"""Video -> ByteTrack -> quality-ranked crops -> OCR/QR -> catalogue -> CSV/coverage."""

import csv
import gc
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

import hydra
from omegaconf import OmegaConf
from PIL import Image

from .build_datasets import checksum, resolve, write_json
from .cpu_ocr import FIELDS, catalogue, match_catalog, valid_gtin
from .crop_quality import quality, rejection_reasons
from .data_loader import code_version, download_data_with_dvc
from .field_extraction import CONTENT_FIELDS, compatible_observations, coverage, crop_fields, fuse
from .text_prototype import TextReader


def bbox_iou(left, right):
    width = max(0, min(left[2], right[2]) - max(left[0], right[0]))
    height = max(0, min(left[3], right[3]) - max(left[1], right[1]))
    intersection = width * height
    left_area = max(0, left[2] - left[0]) * max(0, left[3] - left[1])
    right_area = max(0, right[2] - right[0]) * max(0, right[3] - right[1])
    return intersection / max(1, left_area + right_area - intersection)


def select_crops(cfg, output):
    import cv2
    import torch
    from ultralytics import YOLO

    torch.set_num_threads(cfg.ocr.cpu_threads)
    cv2.setNumThreads(cfg.ocr.cpu_threads)
    classifier, transform = None, None
    if cfg.efficient_weights:
        import timm
        from torchvision import transforms

        checkpoint = torch.load(
            resolve(cfg.efficient_weights), map_location="cpu", weights_only=True
        )
        if checkpoint.get("color") != "RGB" or checkpoint.get("positive_class") != 1:
            raise ValueError("Use a new RGB EfficientNet checkpoint with preprocessing metadata")
        classifier = timm.create_model(checkpoint["architecture"], pretrained=False, num_classes=1)
        classifier.load_state_dict(checkpoint["state_dict"])
        classifier.eval()  # Optional CPU gate, avoiding competition for detector GPU memory.
        transform = transforms.Compose(
            [
                transforms.Resize((checkpoint["size"], checkpoint["size"])),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    videos = sorted(resolve(cfg.videos).glob(cfg.pattern))
    if not videos or not resolve(cfg.yolo_weights).is_file():
        raise ValueError("Local videos or detector weights not found")
    all_tracks = []
    counts = Counter()
    with (output / "filter_audit.jsonl").open("w", encoding="utf-8") as audit:
        for video in videos:
            detector = YOLO(str(resolve(cfg.yolo_weights)))  # Reset tracker between videos.
            capture = cv2.VideoCapture(str(video))
            fps = capture.get(cv2.CAP_PROP_FPS)
            if not capture.isOpened() or not math.isfinite(fps) or fps <= 0:
                raise ValueError(f"Cannot decode video: {video}")
            candidates, observed = defaultdict(list), set()
            states = {}
            next_track = 0
            frame_index = -1
            try:
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    frame_index += 1
                    if cfg.max_frames and frame_index >= cfg.max_frames:
                        break
                    if frame_index % cfg.stride:
                        continue
                    counts["sampled_frames"] += 1
                    prediction = detector.track(
                        frame,
                        persist=True,
                        tracker=cfg.tracker,
                        conf=cfg.detector_conf,
                        iou=0.5,
                        imgsz=cfg.imgsz,
                        device=cfg.detector_device,
                        verbose=False,
                    )[0]
                    for index, box in enumerate(prediction.boxes):
                        counts["detections"] += 1
                        if box.id is None:
                            counts["unconfirmed_detections"] += 1
                            continue
                        tracker_id = int(box.id.item())
                        frame_height, frame_width = frame.shape[:2]
                        x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
                        x1, y1 = max(0, math.floor(x1)), max(0, math.floor(y1))
                        x2, y2 = min(frame_width, math.ceil(x2)), min(frame_height, math.ceil(y2))
                        if x2 <= x1 or y2 <= y1:
                            counts["invalid_bbox"] += 1
                            continue
                        timestamp = frame_index / fps * 1000
                        bbox = [x1, y1, x2, y2]
                        previous = states.get(tracker_id)
                        if (
                            previous is None
                            or timestamp - previous["last"] > cfg.track_max_gap_ms
                            or timestamp - previous["start"] > cfg.track_max_span_ms
                            or bbox_iou(bbox, previous["bbox"]) < cfg.track_min_iou
                        ):
                            next_track += 1
                            previous = dict(track=next_track, start=timestamp)
                            counts["identity_segments"] += 1
                        previous.update(last=timestamp, bbox=bbox)
                        states[tracker_id] = previous
                        track = previous["track"]
                        observed.add(track)
                        crop = Image.fromarray(cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB))
                        values = quality(crop)
                        reasons = rejection_reasons(values, cfg.quality)
                        classifier_score = None
                        if classifier is not None and not reasons:
                            with torch.inference_mode():
                                classifier_score = float(
                                    torch.sigmoid(classifier(transform(crop).unsqueeze(0))).item()
                                )
                            if classifier_score < cfg.efficient_threshold:
                                reasons.append("efficientnet")
                        record = dict(
                            filename=video.name,
                            track_id=track,
                            frame_index=frame_index,
                            frame_timestamp=frame_index / fps * 1000,
                            bbox_xyxy=[x1, y1, x2, y2],
                            quality=values,
                            detector_score=float(box.conf.item()),
                            rejected=reasons,
                            classifier_score=classifier_score,
                        )
                        audit.write(json.dumps(record) + "\n")
                        if reasons:
                            counts["rejected_crops"] += 1
                            for reason in reasons:
                                counts["reject_" + reason] += 1
                            continue
                        counts["accepted_crops"] += 1
                        rank = math.log1p(values["sharpness"]) * math.sqrt(values["min_side"])
                        record["rank"] = rank
                        # Store only top K crops, not every frame of a track.
                        entries = candidates[track]
                        if len(entries) >= cfg.top_k and rank <= entries[-1][0]["rank"]:
                            continue
                        entries.append((record, crop))
                        entries.sort(key=lambda item: item[0]["rank"], reverse=True)
                        del entries[cfg.top_k :]
                for track in sorted(observed):
                    selected = []
                    for slot, (record, crop) in enumerate(candidates[track]):
                        relative = Path("crops") / video.stem / f"track_{track:06d}_{slot}.png"
                        path = output / relative
                        path.parent.mkdir(parents=True, exist_ok=True)
                        crop.save(path)
                        record["image"] = relative.as_posix()
                        record["sha256"] = checksum(path)
                        selected.append(record)
                    all_tracks.append(dict(filename=video.name, track_id=track, crops=selected))
                counts["tracks"] += len(observed)
                counts["tracks_without_usable_crop"] += sum(
                    not candidates[track_key] for track_key in observed
                )
                print(f"Detection complete: {video.name}, {len(observed)} tracks", flush=True)
            finally:
                capture.release()
            del detector
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    write_json(output / "tracks.json", all_tracks)
    write_json(output / "detection_summary.json", dict(counts))
    with (output / "ocr_inputs.jsonl").open("w", encoding="utf-8") as stream:
        for track in all_tracks:
            for crop in track["crops"]:
                stream.write(json.dumps(crop) + "\n")
    return all_tracks, counts


def recognize(cfg, output, tracks):
    reader = TextReader(cfg.ocr)
    fallback = None
    fallback_count = 0
    connection = None
    if cfg.catalogue_csv:
        connection = catalogue(resolve(cfg.catalogue_csv), resolve(cfg.catalogue_db), cfg.encoding)
    rows, per_video = [], defaultdict(list)
    with (output / "details.jsonl").open("w", encoding="utf-8") as details:
        for track in tracks:
            if not track["crops"]:
                continue
            observations = []
            seen = set()
            for crop in track["crops"]:
                if crop["sha256"] in seen:
                    continue  # Identical pixels must not supply independent votes.
                seen.add(crop["sha256"])
                path = (output / crop["image"]).resolve()
                path.relative_to(output.resolve())
                with Image.open(path) as original:
                    image = original.convert("RGB").rotate(cfg.rotation_ccw, expand=True)
                text, truncated = reader.read(image)
                fields, provenance = crop_fields(
                    image,
                    text,
                    reader.last_lines,
                    reader.last_size,
                    cfg.price_template_field,
                    cfg.ocr_confidence,
                )
                observations.append(
                    dict(
                        image=crop["image"],
                        fields=fields,
                        provenance=provenance,
                        text=text,
                        truncated=truncated,
                    )
                )
            best = track["crops"][0]
            compatible, excluded = compatible_observations(observations)
            fields, sources, conflicts = fuse(compatible)
            fallback_text = None
            if (
                cfg.vlm_fallback
                and fallback_count < cfg.vlm_limit
                and (
                    cfg.vlm_name_override
                    or not fields.get("product_name")
                    or not fields.get("price_card")
                )
            ):
                if fallback is None:
                    fallback_cfg = OmegaConf.merge(
                        cfg.ocr, dict(backend=cfg.vlm_backend, device="cuda")
                    )
                    fallback = TextReader(fallback_cfg)
                with Image.open(output / best["image"]) as original:
                    image = original.convert("RGB").rotate(cfg.rotation_ccw, expand=True)
                fallback_text, truncated = fallback.read(image)
                fallback_count += 1
                proposed, _ = crop_fields(image, fallback_text, [], image.size, "", 0.5)
                # Never overwrite decoded/consensus fields; fallback suggestions remain traceable.
                for key, value in proposed.items():
                    if (
                        (not fields.get(key) or (key == "product_name" and cfg.vlm_name_override))
                        and key not in conflicts
                        and not truncated
                    ):
                        fields[key] = value
                        sources[key] = ["vlm_fallback:" + best["image"]]
            matched, suggestions = None, []
            if connection:
                codes = {
                    fields[key]
                    for key in ("barcode", "qr_code_barcode")
                    if valid_gtin(fields.get(key, ""))
                }
                matched, suggestions = match_catalog(
                    connection, codes, fields.get("product_name", ""), 5
                )
                if matched:
                    fields["product_name"] = matched["fullname"]
                    sources["product_name"] = ["catalogue_exact_code"]
            row = dict.fromkeys(FIELDS, "")
            row.update(fields)
            row.update(filename=best["filename"], frame_timestamp=best["frame_timestamp"])
            row.update(dict(zip(["x_min", "y_min", "x_max", "y_max"], best["bbox_xyxy"])))
            rows.append(row)
            per_video[row["filename"]].append(row)
            details.write(
                json.dumps(
                    dict(
                        track_id=track["track_id"],
                        filename=track["filename"],
                        fields=row,
                        observations=observations,
                        field_sources=sources,
                        conflicts=conflicts,
                        excluded_identity_crops=excluded,
                        catalogue_exact=matched,
                        catalogue_suggestions=suggestions,
                        fallback_text=fallback_text,
                    ),
                    ensure_ascii=False,
                )
                + "\n"
            )
            details.flush()
            print(f"OCR {len(rows)}/{len(tracks)}: {track['filename']}", flush=True)
    if connection:
        connection.close()
    with (output / "submission.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    metrics = coverage(rows)
    metrics["core_fields"] = coverage(rows, CONTENT_FIELDS[:12])
    metrics["qr_fields"] = coverage(rows, CONTENT_FIELDS[12:])
    metrics["per_video"] = {video: coverage(values) for video, values in per_video.items()}
    metrics["tracks_detected"] = len(tracks)
    metrics["tracks_with_output"] = len(rows)
    metrics["track_retention"] = len(rows) / len(tracks) if tracks else 0
    denominator = len(tracks) * len(metrics["fields"])
    metrics["coverage_all_detected_tracks"] = (
        metrics["filled_cells"] / denominator if denominator else 0
    )
    metrics["vlm_fallback_calls"] = fallback_count
    metrics["identity_note"] = (
        "One row per bounded ByteTrack segment; fragmentation can duplicate physical tags."
    )
    write_json(output / "metrics.json", metrics)
    return metrics


@hydra.main(version_base=None, config_path="../configs", config_name="video_to_csv")
def main(cfg):
    download_data_with_dvc(cfg.get("dvc_targets", []), cfg.get("pull_dvc", False))
    if cfg.stride < 1 or cfg.top_k < 1 or cfg.max_frames < 0 or cfg.vlm_limit < 0:
        raise ValueError("Invalid frame, crop or fallback budget")
    if cfg.rotation_ccw not in (0, 90, 180, 270):
        raise ValueError("Invalid rotation")
    if cfg.price_template_field not in ("", "price_card", "price_default"):
        raise ValueError("Invalid template price assignment")
    output = resolve(cfg.output)
    output.mkdir(parents=True, exist_ok=False)
    OmegaConf.save(cfg, output / "config.yaml")
    tick = time.perf_counter()
    if cfg.resume_tracks:
        manifest = resolve(cfg.resume_tracks)
        original_root = manifest.parent
        tracks = json.loads(manifest.read_text(encoding="utf-8"))
        # Reuse bounded local crops without repeating detector execution.
        import shutil

        for name in ("detection_summary.json", "filter_audit.jsonl", "ocr_inputs.jsonl"):
            artifact = original_root / name
            if artifact.is_file():
                shutil.copyfile(artifact, output / name)
        write_json(
            output / "detection_source.json",
            dict(
                tracks_sha256=checksum(manifest),
                source_config=(
                    OmegaConf.to_container(
                        OmegaConf.load(original_root / "config.yaml"), resolve=True
                    )
                    if (original_root / "config.yaml").is_file()
                    else None
                ),
            ),
        )
        for track in tracks:
            for crop in track["crops"]:
                source = (original_root / crop["image"]).resolve()
                source.relative_to(original_root.resolve())
                target = (output / crop["image"]).resolve()
                target.relative_to(output.resolve())
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
        write_json(output / "tracks.json", tracks)
    else:
        tracks, _ = select_crops(cfg, output)
    if cfg.detection_only:
        print(f"Detection-only completed: {len(tracks)} track segments")
        return
    metrics = recognize(cfg, output, tracks)
    metrics["total_seconds"] = time.perf_counter() - tick
    metrics["timing_scope"] = "OCR with reused detections" if cfg.resume_tracks else "video to CSV"
    metrics["yolo_sha256"] = checksum(resolve(cfg.yolo_weights))
    metrics["code_version"] = code_version()
    write_json(output / "metrics.json", metrics)
    print(
        json.dumps(
            {
                key: metrics[key]
                for key in (
                    "rows",
                    "coverage",
                    "mean_filled_fields",
                    "coverage_all_detected_tracks",
                    "total_seconds",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
