"""Regroup original annotations by video parsed from filenames."""

import json
import math
import re
import shutil
from collections import Counter
from pathlib import Path

import hydra
from PIL import Image

from .build_datasets import checksum, resolve, write_json


@hydra.main(version_base=None, config_path="../configs", config_name="import_yolo")
def main(cfg):
    source, output = resolve(cfg.source), resolve(cfg.output)
    if output.exists():
        raise FileExistsError(output)
    rows, seen = [], {}
    for split in ("train", "valid", "test"):
        for image in sorted((source / split / "images").iterdir()):
            if image.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            matches = [
                video
                for video in cfg.videos
                if re.search(r"(?:^|__)" + re.escape(video) + r"_", image.name)
            ]
            if len(matches) != 1:
                raise ValueError(f"Ambiguous video: {image.name}")
            video = matches[0]
            label = source / split / "labels" / (image.stem + ".txt")
            if not label.is_file():
                raise FileNotFoundError(label)
            annotations = label.read_text().splitlines()
            for line in annotations:
                values = [float(value) for value in line.split()]
                if (
                    len(values) != 5
                    or values[0] != 0
                    or not all(math.isfinite(value) for value in values)
                ):
                    raise ValueError(f"Invalid YOLO row: {label}: {line}")
                _, center_x, center_y, width, height = values
                if not (
                    0 <= center_x <= 1 and 0 <= center_y <= 1 and 0 < width <= 1 and 0 < height <= 1
                ):
                    raise ValueError(f"Invalid normalized bbox: {label}")
            with Image.open(image) as pixels:
                pixels.verify()
            digest = checksum(image)
            if digest in seen:
                previous = seen[digest]
                if previous["video"] != video or previous["label_sha256"] != checksum(label):
                    raise ValueError(f"Conflicting duplicate: {image}")
                continue
            relative = Path("yolo/images") / video / image.name
            row = dict(
                image=relative.as_posix(),
                video=video + ".mp4",
                fold=list(cfg.videos).index(video),
                boxes=len(annotations),
                source="original_annotation",
                original_split=split,
                sha256=digest,
                label_sha256=checksum(label),
            )
            seen[digest] = row
            rows.append(row)
            target = output / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image, target)
            destination = output / "yolo/labels" / video / label.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(label, destination)
    if {row["fold"] for row in rows} != set(range(len(cfg.videos))):
        raise ValueError("Not all configured videos found")
    write_json(output / "yolo/manifest.json", rows)
    folds = []
    for fold in range(len(cfg.videos)):
        folds.append(
            dict(
                fold=fold,
                train=[row["image"] for row in rows if row["fold"] != fold],
                val=[row["image"] for row in rows if row["fold"] == fold],
            )
        )
    write_json(output / "yolo/folds.json", folds)
    shutil.copytree(resolve(cfg.efficient), output / "efficient")
    efficient_rows = json.loads((output / "efficient/manifest.json").read_text())
    for name in ("README.dataset.txt", "README.roboflow.txt"):
        shutil.copy2(source / name, output / name)
    report = dict(
        yolo_images=len(rows),
        yolo_boxes=sum(row["boxes"] for row in rows),
        by_video=dict(Counter(row["video"] for row in rows)),
        efficient_images=len(efficient_rows),
        groups={video: index for index, video in enumerate(cfg.videos)},
        yolo_labels="original supplied annotations",
        efficient_labels="teacher pseudo-labels",
    )
    write_json(output / "summary.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
