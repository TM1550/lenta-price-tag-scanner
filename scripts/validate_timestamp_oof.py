"""Run fold-specific detector and existing OCR pipeline on annotated timestamps."""

import json
from pathlib import Path

import cv2
import torch
from hydra import compose, initialize_config_module
from PIL import Image
from ultralytics import YOLO

from lenta_price_tag_scanner.build_datasets import checksum
from lenta_price_tag_scanner.evaluate_fields import number, read_rows, score
from lenta_price_tag_scanner.video_to_csv import recognize


def annotated_frames(video, timestamps, tolerance_ms=100):
    """Decode in presentation order, avoiding unreliable average-FPS seeks on VFR."""
    capture = cv2.VideoCapture(str(video))
    previous = None
    current = None
    ended = False
    try:
        for timestamp in timestamps:
            while not ended and (current is None or current[0] < timestamp):
                success, frame = capture.read()
                if not success:
                    ended = True
                    break
                previous = current
                current = (capture.get(cv2.CAP_PROP_POS_MSEC), frame)
            candidates = [item for item in (previous, current) if item is not None]
            nearest = (
                min(candidates, key=lambda item: abs(item[0] - timestamp)) if candidates else None
            )
            if nearest is None or abs(nearest[0] - timestamp) > tolerance_ms:
                yield timestamp, None, None
            else:
                yield timestamp, nearest[0], nearest[1]
    finally:
        capture.release()


def main(root, vlm=False):
    root = Path(root)
    output = root / "outputs/timestamp_oof"
    output.mkdir(parents=True, exist_ok=True)
    folds = json.loads((root / "project/data/oof_v2/yolo/folds.json").read_text())
    torch.set_num_threads(4)
    cv2.setNumThreads(4)
    reports = {}
    with initialize_config_module(config_module="configs", version_base=None):
        cfg = compose(config_name="video_to_csv")
    cfg.catalogue_csv = str(root / "data/db_hack.csv")
    cfg.catalogue_db = str(output / "catalogue.sqlite")
    cfg.ocr.backend = "paddle"
    cfg.ocr.device = "cpu"
    cfg.ocr.paddle_cache = str(root / "audit/source/models/paddle")
    cfg.vlm_fallback = vlm
    cfg.vlm_name_override = True
    cfg.vlm_limit = 100000
    for label in sorted((root / "data").glob("*.csv")):
        video = label.with_suffix(".mp4")
        if not video.exists():
            continue
        chosen = None
        for fold in folds:
            validation = {
                Path(path).parent.name for path in fold.get("val", fold.get("validation", []))
            }
            if video.stem in validation:
                chosen = fold
                break
        status = "oof" if chosen is not None else "external_holdout"
        chosen = chosen or folds[0]
        training = sorted({Path(path).parent.name for path in chosen["train"]})
        if video.stem in training:
            raise ValueError(f"Leakage: {video.stem}")
        checkpoint = (
            root / f"project/runs/stability_yolo/fold_{chosen['fold']}/training/weights/last.pt"
        )
        detector = YOLO(str(checkpoint))
        destination = output / video.stem
        destination.mkdir(exist_ok=True)
        truth = read_rows(label)
        tracks = []
        unavailable = []
        frame_audit = []
        timestamps = sorted({number(row["frame_timestamp"]) for row in truth})
        for timestamp, actual_timestamp, frame in annotated_frames(video, timestamps):
            if frame is None:
                unavailable.append(timestamp)
                continue  # Keep these GT rows in denominator: unavailable input is not success.
            frame_audit.append(dict(requested_ms=timestamp, decoded_ms=actual_timestamp))
            prediction = detector.predict(
                frame, device="cpu", imgsz=1280, conf=0.25, verbose=False
            )[0]
            for box in prediction.boxes.xyxy.tolist():
                left, top, right, bottom = [int(value) for value in box]
                crop = frame[max(0, top) : bottom, max(0, left) : right]
                if not crop.size:
                    continue
                relative = f"crop_{len(tracks):06d}.png"
                Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).save(destination / relative)
                record = dict(
                    filename=video.name,
                    frame_timestamp=timestamp,
                    bbox_xyxy=[left, top, right, bottom],
                    image=relative,
                    sha256=checksum(destination / relative),
                )
                tracks.append(dict(filename=video.name, track_id=len(tracks), crops=[record]))
        (destination / "tracks.json").write_text(json.dumps(tracks))
        recognize(cfg, destination, tracks)
        result = score(truth, read_rows(destination / "submission.csv"))
        result.update(
            status=status,
            fold=chosen["fold"],
            train_videos=training,
            checkpoint=str(checkpoint),
            checkpoint_sha256=checksum(checkpoint),
            efficientnet="disabled, as in final pipeline",
            vlm=vlm,
            unavailable_timestamps=unavailable,
            frame_audit=frame_audit,
        )
        reports[video.stem] = result
        (output / "report.json").write_text(json.dumps(reports, indent=2))
        print(video.stem, result["correct"], result["expected"], flush=True)
    for status in ("oof", "external_holdout"):
        subset = [result for result in reports.values() if result["status"] == status]
        expected = sum(result["expected"] for result in subset)
        correct = sum(result["correct"] for result in subset)
        print(status, correct, expected, correct / expected if expected else None, flush=True)
    summary = {}
    for status in ("oof", "external_holdout"):
        subset = [result for result in reports.values() if result["status"] == status]
        expected = sum(result["expected"] for result in subset)
        correct = sum(result["correct"] for result in subset)
        summary[status] = dict(
            correct=correct, expected=expected, metric=correct / expected if expected else None
        )
    (output / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    import fire

    fire.Fire(main)
