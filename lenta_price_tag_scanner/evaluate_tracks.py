"""Link GT and OCR CSV anchors through dense tracker history, without text matching."""

import csv
import html
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import hydra
from omegaconf import OmegaConf
from PIL import Image

from .build_datasets import checksum, resolve, write_json
from .evaluate_fields import META, normalized, number, read_rows
from .video_to_csv import bbox_iou

BBOX = ("x_min", "y_min", "x_max", "y_max")


def json_lines(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def decoder_times(video):
    import cv2

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"Cannot open {video}")
    expected = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    times = []
    try:
        while capture.grab():
            timestamp = float(capture.get(cv2.CAP_PROP_POS_MSEC))
            if not math.isfinite(timestamp) or timestamp < 0 or (times and timestamp <= times[-1]):
                raise ValueError(f"Invalid decoder PTS: {video}, frame {len(times)}")
            times.append(timestamp)
    finally:
        capture.release()
    if not times or (expected > 0 and len(times) != expected):
        raise ValueError(f"Incomplete decode: {video}: {len(times)}/{expected}")
    return times


def associate(
    bbox, timestamp, observations, last_timestamp, threshold=0.5, margin=0.1, tolerance=100
):
    """Choose only the nearest observed frame; no maximizing IoU across a time window."""
    if (
        len(bbox) != 4
        or not all(math.isfinite(coordinate) for coordinate in bbox)
        or bbox[2] <= bbox[0]
        or bbox[3] <= bbox[1]
        or not math.isfinite(timestamp)
    ):
        raise ValueError("Invalid anchor geometry or timestamp")
    if timestamp < 0 or timestamp > last_timestamp:
        return dict(status="timestamp_outside_video", candidates=[])
    if not observations:
        return dict(status="no_track_observations", candidates=[])
    frame = min(observations, key=lambda i: (abs(observations[i][0]["pts_ms"] - timestamp), i))
    observed_time = observations[frame][0]["pts_ms"]
    base = dict(
        frame_index=frame,
        matched_pts_ms=observed_time,
        time_distance_ms=abs(observed_time - timestamp),
    )
    if base["time_distance_ms"] > tolerance:
        return dict(base, status="no_nearby_observed_frame", candidates=[])
    candidates = sorted(
        [
            dict(track_id=row["track_id"], iou=bbox_iou(bbox, row["bbox_xyxy"]))
            for row in observations[frame]
        ],
        key=lambda row: (-row["iou"], row["track_id"]),
    )
    candidates = [row for row in candidates if row["iou"] >= threshold]
    if not candidates:
        return dict(base, status="no_overlapping_track", candidates=[])
    if len(candidates) > 1 and candidates[0]["iou"] - candidates[1]["iou"] < margin:
        return dict(base, status="ambiguous_tracks", candidates=candidates)
    return dict(
        base,
        status="linked",
        track_id=candidates[0]["track_id"],
        iou=candidates[0]["iou"],
        candidates=candidates,
    )


def load_predictions(root, times):
    """Recover exact frame identity from pipeline sidecars rather than legacy FPS times."""
    predictions = read_rows(root / "submission.csv")
    details = json_lines(root / "details.jsonl")
    tracks = json.loads((root / "tracks.json").read_text(encoding="utf-8"))
    by_track = {(row["filename"], int(row["track_id"])): row for row in tracks}
    if len(details) != len(predictions):
        raise ValueError("CSV/details row counts differ; cannot trust positional linkage")
    result = []
    for index, (row, detail) in enumerate(zip(predictions, details)):
        if detail.get("csv_row_index", index) != index:
            raise ValueError("CSV/details indices differ")
        if any(str(detail["fields"].get(key, "")) != str(row.get(key, "")) for key in row):
            raise ValueError(f"CSV/details content mismatch at row {index}")
        video = Path(row["filename"].strip()).stem
        if video not in times:
            continue
        track = by_track[(detail["filename"], int(detail["track_id"]))]
        if not track["crops"]:
            raise ValueError("CSV row has no crop provenance")
        best = track["crops"][0]
        frame = int(best["frame_index"])
        if frame < 0 or frame >= len(times[video]):
            raise ValueError("Prediction frame outside decoded video")
        if any(abs(number(row[key]) - coord) > 1e-5 for key, coord in zip(BBOX, best["bbox_xyxy"])):
            raise ValueError("CSV bbox differs from crop provenance")
        image = (root / best["image"]).resolve()
        if not image.is_relative_to(root.resolve()) or not image.is_file():
            raise ValueError("Invalid crop path")
        result.append(
            dict(
                index=index,
                video=video,
                fields=row,
                frame_index=frame,
                pts_ms=times[video][frame],
                bbox=list(best["bbox_xyxy"]),
                quality_rank=best.get("rank", 0),
                image=image,
                pipeline_track_id=detail["track_id"],
            )
        )
    return result


def link_rows(truth, predictions, histories, times, threshold, margin, tolerance):
    reference_links, prediction_links = [], []
    for target in truth:
        link = associate(
            target["bbox"],
            target["timestamp"],
            histories[target["video"]],
            times[target["video"]][-1],
            threshold,
            margin,
            tolerance,
        )
        reference_links.append(dict(target=target, link=link))
    for prediction in predictions:
        link = associate(
            prediction["bbox"],
            prediction["pts_ms"],
            histories[prediction["video"]],
            times[prediction["video"]][-1],
            threshold,
            margin,
            tolerance,
        )
        prediction_links.append(dict(prediction=prediction, link=link))
    reference_groups, prediction_groups = defaultdict(list), defaultdict(list)
    for entry in reference_links:
        if entry["link"]["status"] == "linked":
            reference_groups[(entry["target"]["video"], entry["link"]["track_id"])].append(entry)
    for entry in prediction_links:
        if entry["link"]["status"] == "linked":
            prediction_groups[(entry["prediction"]["video"], entry["link"]["track_id"])].append(
                entry
            )
    for key, entries in reference_groups.items():
        if len(entries) > 1:
            for entry in entries:
                entry["link"]["status"] = "multiple_gt_anchors_on_track_review_required"
            continue
        candidates = prediction_groups.get(key, [])
        if not candidates:
            entries[0]["link"]["status"] = "track_without_csv_prediction"
            continue
        # Select before reading OCR correctness; never pick the answer closest to GT.
        ranked = sorted(
            candidates,
            key=lambda candidate: (
                -candidate["prediction"]["quality_rank"],
                -candidate["link"]["iou"],
                candidate["prediction"]["index"],
            ),
        )
        entries[0]["prediction"] = ranked[0]["prediction"]
        entries[0]["duplicate_prediction_indices"] = [
            candidate["prediction"]["index"] for candidate in ranked[1:]
        ]
    return reference_links, prediction_links


def compare_fields(reference_links):
    cells, matches = [], []
    totals, per_video, per_field = Counter(), defaultdict(Counter), defaultdict(Counter)
    for entry in reference_links:
        target, link = entry["target"], entry["link"]
        prediction = entry.get("prediction")
        count = Counter()
        for field, expected in target["fields"].items():
            if field in META or not normalized(expected):
                continue
            actual = prediction["fields"].get(field, "") if prediction else ""
            correct = int(bool(prediction) and normalized(actual) == normalized(expected))
            count.update(expected=1, correct=correct)
            per_field[field].update(expected=1, correct=correct)
            cells.append(
                dict(
                    gt_id=target["id"],
                    video=target["video"],
                    field=field,
                    expected=expected,
                    predicted=actual,
                    correct=correct,
                    status=link["status"],
                )
            )
        per_video[target["video"]].update(count)
        totals.update(count)
        matches.append(
            dict(
                gt_id=target["id"],
                video=target["video"],
                gt_timestamp_ms=target["timestamp"],
                status=link["status"],
                reference_track_id=link.get("track_id", ""),
                anchor_frame_index=link.get("frame_index", ""),
                anchor_pts_ms=link.get("matched_pts_ms", ""),
                anchor_iou=link.get("iou", ""),
                prediction_csv_row=prediction["index"] if prediction else "",
                pipeline_track_id=prediction["pipeline_track_id"] if prediction else "",
                prediction_frame_index=prediction["frame_index"] if prediction else "",
                duplicate_prediction_indices=json.dumps(
                    entry.get("duplicate_prediction_indices", [])
                ),
                correct=count["correct"],
                expected=count["expected"],
            )
        )

    def metric(counts):
        return dict(
            counts,
            correct_fill=counts["correct"] / counts["expected"] if counts["expected"] else None,
        )

    without_color = Counter(totals)
    without_color.subtract(per_field.get("color", Counter()))
    prices = Counter()
    for field in ("price_default", "price_card", "price_discount"):
        prices.update(per_field.get(field, Counter()))

    return (
        matches,
        cells,
        dict(
            **metric(totals),
            per_video={key: metric(val) for key, val in per_video.items()},
            per_field={key: metric(val) for key, val in per_field.items()},
            field_groups=dict(without_color=metric(without_color), printed_prices=metric(prices)),
        ),
    )


def write_crop_map(tracking_root, output, matches, times):
    """Keep all selected crops traceable, including unlinked and ambiguous tracks."""
    references = defaultdict(list)
    for match in matches:
        if str(match["reference_track_id"]) != "":
            references[(match["video"], int(match["reference_track_id"]))].append(match)
    tracks = json.loads((tracking_root / "tracks.json").read_text(encoding="utf-8"))
    rows = []
    for track in tracks:
        video = Path(track["filename"]).stem
        if video not in times:
            continue
        candidates = references.get((video, int(track["track_id"])), [])
        unique = len(candidates) == 1 and candidates[0]["status"] in {
            "linked",
            "track_without_csv_prediction",
        }
        for crop in track["crops"]:
            frame = int(crop["frame_index"])
            rows.append(
                dict(
                    video=video,
                    reference_track_id=track["track_id"],
                    raw_tracker_id=crop.get("tracker_id", ""),
                    crop=crop["image"],
                    frame_index=frame,
                    pts_ms=times[video][frame],
                    proposed_gt_id=candidates[0]["gt_id"] if unique else "",
                    candidate_gt_ids=json.dumps([row["gt_id"] for row in candidates]),
                    manual_reviewed=False,
                )
            )
    write_csv(output / "crop_to_reference.csv", rows)


def review(output, videos, reference_links):
    """Show source GT crops and the crop whose OCR contributed the selected CSV row."""
    import cv2

    (output / "review_crops").mkdir()
    requests = defaultdict(lambda: defaultdict(list))
    for entry in reference_links:
        frame = entry["link"].get("frame_index")
        if frame is not None:
            requests[entry["target"]["video"]][frame].append(entry)
    for video in videos:
        pending = requests[video.stem]
        if not pending:
            continue
        capture = cv2.VideoCapture(str(video))
        index = -1
        try:
            while pending:
                ok, pixels = capture.read()
                if not ok:
                    break
                index += 1
                for entry in pending.pop(index, []):
                    x1, y1, x2, y2 = entry["target"]["bbox"]
                    frame_height, frame_width = pixels.shape[:2]
                    crop = pixels[
                        max(0, math.floor(y1)) : min(frame_height, math.ceil(y2)),
                        max(0, math.floor(x1)) : min(frame_width, math.ceil(x2)),
                    ]
                    if crop.size:
                        name = f"review_crops/{entry['target']['id']}_gt.jpg"
                        image = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).rotate(
                            90, expand=True
                        )
                        image.thumbnail((600, 400))
                        image.save(output / name)
                        entry["gt_image"] = name
        finally:
            capture.release()
    cards = []
    for entry in reference_links:
        target, link, prediction = entry["target"], entry["link"], entry.get("prediction")
        images = ""
        if entry.get("gt_image"):
            images += f'<figure><img src="{entry["gt_image"]}"><figcaption>GT anchor crop</figcaption></figure>'
        if prediction:
            name = f"review_crops/{target['id']}_prediction.jpg"
            with Image.open(prediction["image"]) as image:
                image = image.convert("RGB").rotate(90, expand=True)
                image.thumbnail((600, 400))
                image.save(output / name)
            images += f'<figure><img src="{name}"><figcaption>CSV source crop</figcaption></figure>'
        values = dict(
            gt=target["fields"], prediction=prediction["fields"] if prediction else {}, link=link
        )
        cards.append(
            f'<article><h2>{html.escape(target["id"])}: {html.escape(link["status"])}</h2>'
            f'<div class="pair">{images}</div><details><summary>Fields and geometry</summary>'
            f"<pre>{html.escape(json.dumps(values, ensure_ascii=False, indent=2))}</pre></details></article>"
        )
    (output / "review.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Automatic track associations — review required</title>'
        "<style>body{font:16px sans-serif;background:#eee}article{background:white;padding:16px;margin:16px}"
        ".pair{display:flex;flex-wrap:wrap}img{max-width:100%;max-height:400px}pre{white-space:pre-wrap}</style>"
        "<h1>Automatic associations: verify same physical price tag</h1>" + "".join(cards),
        encoding="utf-8",
    )


@hydra.main(version_base=None, config_path="../configs", config_name="evaluate_tracks")
def main(cfg):
    import cv2

    if not 0 <= cfg.iou_threshold <= 1 or cfg.ambiguity_margin < 0 or cfg.time_tolerance_ms < 0:
        raise ValueError("Invalid matching thresholds")
    root, prediction_root, output = (
        resolve(cfg.tracking_run),
        resolve(cfg.prediction_run),
        resolve(cfg.output),
    )
    if output.exists():
        raise FileExistsError(output)
    videos = sorted(resolve(cfg.videos).glob(cfg.pattern))
    if not videos:
        raise ValueError("No videos found")
    cv2.setNumThreads(cfg.cpu_threads)
    times = {video.stem: decoder_times(video) for video in videos}
    histories = {video.stem: defaultdict(list) for video in videos}
    for observation in json_lines(root / "filter_audit.jsonl"):
        video = Path(observation["filename"]).stem
        if video not in times:
            continue
        frame = int(observation["frame_index"])
        if not 0 <= frame < len(times[video]):
            raise ValueError("Tracker observation frame outside video")
        observation["pts_ms"] = times[video][frame]
        histories[video][frame].append(observation)
    truth = []
    for video in videos:
        for index, row in enumerate(read_rows(video.with_suffix(".csv"))):
            truth.append(
                dict(
                    id=f"{video.stem}_{index:04d}",
                    video=video.stem,
                    fields=row,
                    timestamp=number(row["frame_timestamp"]),
                    bbox=[number(row[key]) for key in BBOX],
                )
            )
    predictions = load_predictions(prediction_root, times)
    links, predicted_links = link_rows(
        truth,
        predictions,
        histories,
        times,
        cfg.iou_threshold,
        cfg.ambiguity_margin,
        cfg.time_tolerance_ms,
    )
    matches, cells, report = compare_fields(links)
    output.mkdir(parents=True)
    OmegaConf.save(cfg, output / "config.yaml")
    write_csv(output / "gt_to_prediction.csv", matches)
    write_csv(output / "field_comparison.csv", cells)
    write_crop_map(root, output, matches, times)
    selected_for = {
        row["prediction_csv_row"]: row["gt_id"]
        for row in matches
        if row["prediction_csv_row"] != ""
    }
    write_json(
        output / "prediction_to_track.json",
        [
            dict(
                csv_row=entry["prediction"]["index"],
                video=entry["prediction"]["video"],
                pipeline_track_id=entry["prediction"]["pipeline_track_id"],
                selected_for_gt_id=selected_for.get(entry["prediction"]["index"], ""),
                **entry["link"],
            )
            for entry in predicted_links
        ],
    )
    write_json(output / "frame_times.json", times)
    review(output, videos, links)
    matched = sum("prediction" in entry for entry in links)
    report.update(
        gt_rows=len(truth),
        prediction_rows=len(predictions),
        matched_gt_rows=matched,
        association_statuses=dict(Counter(entry["link"]["status"] for entry in links)),
        duplicate_csv_rows=sum(
            len(entry.get("duplicate_prediction_indices", [])) for entry in links
        ),
        predictions_not_selected=len(predictions) - matched,
        automatic_associations=True,
        manual_identity_review_completed=False,
        oof=False,
        tracking_config=(
            OmegaConf.to_container(OmegaConf.load(root / "config.yaml"), resolve=True)
            if (root / "config.yaml").is_file()
            else None
        ),
        tracking_audit_sha256=checksum(root / "filter_audit.jsonl"),
        predictions_sha256=checksum(prediction_root / "submission.csv"),
        video_sha256={video.name: checksum(video) for video in videos},
        gt_csv_sha256={video.name: checksum(video.with_suffix(".csv")) for video in videos},
        note="Diagnostic automatic track association, not official or independent OOF accuracy. "
        "GT/prediction anchors matched by geometry and actual decoder PTS; OCR values do not select pairs. "
        "Unmatched and ambiguous GT fields remain in denominator. Extra predictions are reported separately: GT is partial.",
    )
    write_json(output / "report.json", report)
    (output / "COMPLETE.txt").write_text(
        "Automatic association and field scoring completed.\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                key: value
                for key, value in report.items()
                if key not in {"per_field", "tracking_config"}
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
