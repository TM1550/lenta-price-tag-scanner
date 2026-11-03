"""Video-grouped teacher labels, independent of questionable CSV timestamps."""

import json
from pathlib import Path

import cv2
import hydra
import torch
from omegaconf import OmegaConf
from PIL import Image
from ultralytics import YOLO

from .build_datasets import checksum
from .crop_quality import quality, rejection_reasons


@hydra.main(version_base=None, config_path="../configs", config_name="pseudo_labels")
def main(cfg):
    output = Path(cfg.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    videos = sorted(Path(cfg.videos).resolve().glob("*.mp4"))
    if len(videos) < 2:
        raise ValueError("At least two videos required for grouped folds")
    model = YOLO(cfg.weights)
    torch.set_num_threads(cfg.threads)
    cv2.setNumThreads(cfg.threads)
    output.mkdir(parents=True)
    OmegaConf.save(cfg, output / "config.yaml")
    teacher = checksum(Path(cfg.weights))
    manifest = []
    counts = {}
    with (output / "audit.jsonl").open("w", encoding="utf-8") as audit:
        for fold_index, video in enumerate(videos):
            capture = cv2.VideoCapture(str(video))
            if not capture.isOpened():
                raise ValueError(video)
            next_ms = 0
            frame_index = -1
            accepted = 0
            while True:
                success, frame = capture.read()
                if not success:
                    break
                frame_index += 1
                timestamp = capture.get(cv2.CAP_PROP_POS_MSEC)
                if timestamp < next_ms:
                    continue
                next_ms = timestamp + cfg.interval_ms
                result = model.predict(
                    frame,
                    device="cpu",
                    conf=cfg.proposal_conf,
                    iou=cfg.iou,
                    imgsz=cfg.imgsz,
                    verbose=False,
                )[0]
                height, width = frame.shape[:2]
                boxes, reasons = [], []
                for detection in result.boxes:
                    left, top, right, bottom = detection.xyxy[0].tolist()
                    confidence = float(detection.conf.item())
                    crop = frame[
                        max(0, int(top)) : min(height, int(bottom)),
                        max(0, int(left)) : min(width, int(right)),
                    ]
                    if not crop.size:
                        reasons.append("empty_crop")
                        continue
                    values = quality(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
                    rejected = rejection_reasons(values, cfg.quality)
                    if confidence < cfg.accept_conf:
                        rejected.append("confidence")
                    reasons.extend(rejected)
                    boxes.append(
                        dict(bbox=[left, top, right, bottom], confidence=confidence, quality=values)
                    )
                if not boxes:
                    reasons.append("no_detection")
                audit.write(
                    json.dumps(
                        dict(
                            video=video.name,
                            frame=frame_index,
                            timestamp_ms=timestamp,
                            reasons=reasons,
                            detections=boxes,
                        )
                    )
                    + "\n"
                )
                if reasons:
                    continue
                stem = f"{video.stem}_{frame_index:07d}"
                image = output / "yolo/images" / video.stem / (stem + ".jpg")
                label = output / "yolo/labels" / video.stem / (stem + ".txt")
                image.parent.mkdir(parents=True, exist_ok=True)
                label.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(image), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                    raise IOError(image)
                lines = []
                for box in boxes:
                    left, top, right, bottom = box["bbox"]
                    lines.append(
                        f"0 {(left+right)/2/width:.8f} {(top+bottom)/2/height:.8f} "
                        f"{(right-left)/width:.8f} {(bottom-top)/height:.8f}"
                    )
                label.write_text("\n".join(lines) + "\n")
                manifest.append(
                    dict(
                        video=video.stem,
                        fold=fold_index,
                        frame=frame_index,
                        timestamp_ms=timestamp,
                        image=image.relative_to(output).as_posix(),
                        label=label.relative_to(output).as_posix(),
                        boxes=boxes,
                        source="teacher_pseudo",
                        reviewed=False,
                    )
                )
                accepted += 1
            capture.release()
            counts[video.stem] = accepted
            print(video.name, accepted, flush=True)
    folds = []
    for fold, video in enumerate(videos):
        train = [row["image"] for row in manifest if row["video"] != video.stem]
        val = [row["image"] for row in manifest if row["video"] == video.stem]
        folds.append(
            dict(
                fold=fold,
                validation_video=video.stem,
                train=train,
                val=val,
                usable=bool(train and val),
            )
        )
    (output / "yolo").mkdir(exist_ok=True)
    (output / "yolo/manifest.json").write_text(json.dumps(manifest, indent=2))
    (output / "yolo/folds.json").write_text(json.dumps(folds, indent=2))
    (output / "summary.json").write_text(
        json.dumps(
            dict(
                frames=counts,
                teacher_sha256=teacher,
                total_frames=len(manifest),
                boxes=sum(len(row["boxes"]) for row in manifest),
                note="Pseudo labels, not human ground truth. Teacher training provenance unverified.",
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
