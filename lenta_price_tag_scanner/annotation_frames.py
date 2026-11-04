"""Prepare unlabeled full-resolution frames with video-grouped fold provenance."""

import hashlib
import json
import math
import zipfile
from pathlib import Path

import cv2
import hydra
from omegaconf import OmegaConf
from PIL import Image, ImageDraw


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@hydra.main(version_base=None, config_path="../configs", config_name="annotation_frames")
def main(cfg):
    videos = sorted(Path(cfg.videos).resolve().glob("*.mp4"))
    output = Path(cfg.output).resolve()
    if len(videos) < 2:
        raise ValueError("At least two source videos required")
    if not 0 < cfg.candidate_step_ms <= cfg.interval_ms:
        raise ValueError("Require 0 < candidate_step_ms <= interval_ms")
    output.mkdir(parents=True, exist_ok=False)
    images = output / "images"
    images.mkdir()
    OmegaConf.save(cfg, output / "config.yaml")
    cv2.setNumThreads(cfg.threads)
    rows, sources = [], []
    for fold, video in enumerate(videos):
        capture = cv2.VideoCapture(str(video))
        if not capture.isOpened():
            raise ValueError(f"Cannot open {video}")
        expected = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        best, bucket, index, next_ms, previous_ms = None, None, -1, 0.0, -1.0
        thumbnails = []

        def save(candidate):
            if candidate is None:
                return
            score, frame_id, timestamp, frame = candidate
            name = f"{video.stem}__f{frame_id:07d}__t{round(timestamp):09d}ms.jpg"
            target = images / name
            if not cv2.imwrite(str(target), frame, [cv2.IMWRITE_JPEG_QUALITY, cfg.jpeg_quality]):
                raise IOError(target)
            height, width = frame.shape[:2]
            rows.append(
                dict(
                    image=name,
                    video=video.name,
                    fold=fold,
                    frame_index=frame_id,
                    timestamp_ms=timestamp,
                    width=width,
                    height=height,
                    sharpness=score,
                    sha256=sha256(target),
                    annotated=False,
                )
            )
            thumb = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            thumb.thumbnail((320, 180))
            tile = Image.new("RGB", (320, 205), "white")
            tile.paste(thumb, (0, 0))
            ImageDraw.Draw(tile).text(
                (3, 184), f"f{frame_id} | {timestamp / 1000:.2f}s", fill="black"
            )
            thumbnails.append(tile)

        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                index += 1
                timestamp = float(capture.get(cv2.CAP_PROP_POS_MSEC))
                if (
                    not math.isfinite(timestamp)
                    or timestamp < 0
                    or (index and timestamp <= previous_ms)
                ):
                    raise ValueError(f"Non-increasing decoder timestamp: {video}, frame {index}")
                previous_ms = timestamp
                current_bucket = int(timestamp // cfg.interval_ms)
                if current_bucket != bucket:
                    save(best)
                    best, bucket = None, current_bucket
                if timestamp < next_ms:
                    continue
                next_ms = timestamp + cfg.candidate_step_ms
                gray = cv2.cvtColor(cv2.resize(frame, (960, 540)), cv2.COLOR_BGR2GRAY)
                score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                if best is None or score > best[0]:
                    best = (score, index, timestamp, frame.copy())
            save(best)
        finally:
            capture.release()
        if expected > 0 and index + 1 != expected:
            raise ValueError(f"Incomplete decode {video}: {index + 1}/{expected}")
        for page, start in enumerate(range(0, len(thumbnails), 40)):
            tiles = thumbnails[start : start + 40]
            sheet = Image.new("RGB", (1280, math.ceil(len(tiles) / 4) * 205), "white")
            for j, tile in enumerate(tiles):
                sheet.paste(tile, ((j % 4) * 320, (j // 4) * 205))
            sheet.save(output / f"preview_{video.stem}_{page + 1}.jpg", quality=85)
        sources.append(
            dict(
                video=video.name,
                sha256=sha256(video),
                fold=fold,
                decoded_frames=index + 1,
                last_timestamp_ms=previous_ms,
                selected_frames=len(thumbnails),
            )
        )
        print(json.dumps(sources[-1]), flush=True)
    (output / "manifest.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (output / "sources.json").write_text(json.dumps(sources, indent=2), encoding="utf-8")
    folds = {
        str(i): {
            "val_videos": [v.name],
            "train_videos": [other.name for other in videos if other != v],
        }
        for i, v in enumerate(videos)
    }
    (output / "folds.json").write_text(json.dumps(folds, indent=2), encoding="utf-8")
    with zipfile.ZipFile(output / "roboflow_images.zip", "w", zipfile.ZIP_STORED) as archive:
        for path in sorted(images.glob("*.jpg")):
            archive.write(path, path.name)
    (output / "COMPLETE.txt").write_text(
        f"Prepared {len(rows)} unlabeled images.\n", encoding="utf-8"
    )
    print(f"Complete: {len(rows)} images in {output}", flush=True)


if __name__ == "__main__":
    main()
