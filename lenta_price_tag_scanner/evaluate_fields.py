"""Conservative spatial/temporal field scoring; no OCR text used for matching."""

import csv
import json
import re
from pathlib import Path

META = {"filename", "frame_timestamp", "x_min", "y_min", "x_max", "y_max"}
EMPTY = {"", "нет", "none", "null", "nan", "n/a"}


def read_rows(path):
    for encoding in ("utf-8-sig", "cp1251"):
        try:
            with Path(path).open(encoding=encoding, newline="") as stream:
                return list(csv.DictReader(stream))
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Cannot decode {path}")


def number(value):
    return float(str(value).replace(",", "."))


def normalized(value):
    text = re.sub(r"\s+", " ", str(value or "").strip().casefold())
    if text in EMPTY:
        return ""
    if re.fullmatch(r"-?\d+[.,]\d+", text):
        return format(number(text), ".10g")
    return text


def overlap(first, second):
    left, top, right, bottom = [number(first[key]) for key in ("x_min", "y_min", "x_max", "y_max")]
    other_left, other_top, other_right, other_bottom = [
        number(second[key]) for key in ("x_min", "y_min", "x_max", "y_max")
    ]
    intersection = max(0, min(right, other_right) - max(left, other_left)) * max(
        0, min(bottom, other_bottom) - max(top, other_top)
    )
    union = (
        max(0, right - left) * max(0, bottom - top)
        + max(0, other_right - other_left) * max(0, other_bottom - other_top)
        - intersection
    )
    return intersection / union if union else 0


def score(truth, predictions, iou=0.3, tolerance_ms=0):
    fields = [key for key in truth[0] if key not in META] if truth else []
    candidates = []
    for target_index, target in enumerate(truth):
        for prediction_index, prediction in enumerate(predictions):
            distance = abs(
                number(target["frame_timestamp"]) - number(prediction["frame_timestamp"])
            )
            if distance <= tolerance_ms:
                similarity = overlap(target, prediction)
                if similarity >= iou:
                    candidates.append((-similarity, distance, target_index, prediction_index))
    matches = {}
    used = set()
    for _, _, target_index, prediction_index in sorted(candidates):
        if target_index not in matches and prediction_index not in used:
            matches[target_index] = prediction_index
            used.add(prediction_index)
    counts = {field: {"correct": 0, "expected": 0} for field in fields}
    for target_index, target in enumerate(truth):
        prediction = predictions[matches[target_index]] if target_index in matches else {}
        for field in fields:
            expected = normalized(target.get(field))
            if expected:
                counts[field]["expected"] += 1
                counts[field]["correct"] += int(normalized(prediction.get(field)) == expected)
    correct = sum(item["correct"] for item in counts.values())
    expected = sum(item["expected"] for item in counts.values())
    return dict(
        correct=correct,
        expected=expected,
        correct_fill=correct / expected if expected else None,
        matched_rows=len(matches),
        gt_rows=len(truth),
        prediction_rows=len(predictions),
        unmatched_predictions=len(predictions) - len(used),
        per_field=counts,
    )


def main(labels, predictions, output, iou=0.3, tolerance_ms=0, oof_manifest=None):
    """Score CSV predictions. OOF manifest lists video, fold, train_videos, checkpoint."""
    predicted = read_rows(predictions)
    results = {}
    evidence = json.loads(Path(oof_manifest).read_text()) if oof_manifest else None
    for csv_path in sorted(Path(labels).glob("*.csv")):
        if not csv_path.with_suffix(".mp4").exists():
            continue
        video = csv_path.stem
        if evidence is not None:
            entry = evidence[video]
            if video in entry["train_videos"] or not entry["checkpoint"] or "fold" not in entry:
                raise ValueError(f"Invalid OOF provenance: {video}")
        rows = [row for row in predicted if Path(row["filename"]).stem == video]
        results[video] = score(read_rows(csv_path), rows, float(iou), float(tolerance_ms))
    if not results:
        raise ValueError("No CSV/video pairs found")
    correct = sum(item["correct"] for item in results.values())
    expected = sum(item["expected"] for item in results.values())
    report = dict(
        per_video=results,
        correct=correct,
        expected=expected,
        micro_correct_fill=correct / expected if expected else None,
        oof_provenance_supplied=evidence is not None,
        iou=iou,
        tolerance_ms=tolerance_ms,
        note="Strict normalized equality on nonempty GT fields; greedy one-to-one geometric matching. OOF manifest is declared provenance, not proof of training isolation. Not an official hackathon scorer.",
    )
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "per_video"}, ensure_ascii=False
        )
    )


if __name__ == "__main__":
    import fire

    fire.Fire(main)
