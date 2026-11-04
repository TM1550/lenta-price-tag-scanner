"""Build and validate portable video-grouped YOLO folds from cleaned annotations."""

import json
import math
import shutil
from collections import Counter
from pathlib import Path

import hydra
from omegaconf import OmegaConf
from PIL import Image

from .build_datasets import checksum, resolve, write_json


def local_path(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Path outside dataset: {relative}")
    return path


def label_count(path, class_count):
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for line in lines:
        values = [float(value) for value in line.split()]
        if len(values) != 5 or not all(math.isfinite(value) for value in values):
            raise ValueError(f"Invalid detection label: {path}")
        cls, x, y, width, height = values
        if (
            cls != int(cls)
            or not 0 <= cls < class_count
            or not (0 < width <= 1 and 0 < height <= 1)
        ):
            raise ValueError(f"Invalid class or dimensions: {path}")
        if not (
            width / 2 - 1e-8 <= x <= 1 - width / 2 + 1e-8
            and height / 2 - 1e-8 <= y <= 1 - height / 2 + 1e-8
        ):
            raise ValueError(f"Box outside image: {path}")
    return len(lines)


def make_folds(rows, expected_folds):
    if {row["fold"] for row in rows} != set(range(expected_folds)):
        raise ValueError("Expected one nonempty group for each fold")
    image_paths = [row["image"] for row in rows]
    if len(image_paths) != len(set(image_paths)):
        raise ValueError("Duplicate image paths")
    video_fold = {}
    for row in rows:
        if video_fold.setdefault(row["video"], row["fold"]) != row["fold"]:
            raise ValueError("A video was assigned to multiple folds")
    if len(video_fold) != expected_folds:
        raise ValueError("This dataset requires exactly one validation video per fold")
    return [
        dict(
            fold=fold,
            train=[row["image"] for row in rows if row["fold"] != fold],
            val=[row["image"] for row in rows if row["fold"] == fold],
        )
        for fold in range(expected_folds)
    ]


def validate_dataset(root, expected_folds=5):
    root = Path(root).resolve()
    rows = json.loads((root / "yolo/manifest.json").read_text(encoding="utf-8"))
    folds = json.loads((root / "yolo/folds.json").read_text(encoding="utf-8"))
    expected = make_folds(rows, expected_folds)
    if folds != expected:
        raise ValueError("Stored folds disagree with manifest")
    names = OmegaConf.load(root / "classes.yaml").names
    hashes = set()
    total_boxes = 0
    for row in rows:
        image, label = local_path(root, row["image"]), local_path(root, row["label"])
        with Image.open(image) as pixels:
            pixels.verify()
        digest = checksum(image)
        if digest != row["sha256"] or digest in hashes:
            raise ValueError(f"Changed or duplicated image: {image}")
        hashes.add(digest)
        if checksum(label) != row["label_sha256"] or label_count(label, len(names)) != row["boxes"]:
            raise ValueError(f"Changed label: {label}")
        total_boxes += row["boxes"]
    coverage, summaries = Counter(), []
    by_image = {row["image"]: row for row in rows}
    for fold in folds:
        training, validation = set(fold["train"]), set(fold["val"])
        train_videos = {by_image[image]["video"] for image in training}
        val_videos = {by_image[image]["video"] for image in validation}
        if (
            not training
            or not validation
            or training & validation
            or training | validation != set(by_image)
            or train_videos & val_videos
            or len(val_videos) != 1
        ):
            raise ValueError("Incomplete or leaking fold")
        folder = root / "folds" / f"fold_{fold['fold']}"
        for split in ("train", "val"):
            paths = (folder / f"{split}.txt").read_text(encoding="utf-8").splitlines()
            resolved = [local_path(root, folder / path) for path in paths]
            if resolved != [local_path(root, path) for path in fold[split]]:
                raise ValueError("Split list disagrees with fold manifest")
        data = OmegaConf.to_container(OmegaConf.load(folder / "data.yaml"))
        if data != dict(train="train.txt", val="val.txt", names=list(names), nc=len(names)):
            raise ValueError("Unexpected fold data.yaml")
        coverage.update(validation)
        summaries.append(
            dict(
                fold=fold["fold"],
                val_video=next(iter(val_videos)),
                train_images=len(training),
                val_images=len(validation),
                train_boxes=sum(by_image[path]["boxes"] for path in training),
                val_boxes=sum(by_image[path]["boxes"] for path in validation),
            )
        )
    if set(coverage) != set(by_image) or set(coverage.values()) != {1}:
        raise ValueError("Each frame must be in validation exactly once")
    return dict(
        images=len(rows),
        boxes=total_boxes,
        folds=summaries,
        video_leakage=False,
        exact_duplicate_images=False,
        validation_coverage_once=True,
        image_and_label_hashes_verified=True,
    )


@hydra.main(version_base=None, config_path="../configs", config_name="build_yolo_folds")
def main(cfg):
    source, output = resolve(cfg.source), resolve(cfg.output)
    if output.exists():
        raise FileExistsError(output)
    if source in output.parents or output in source.parents:
        raise ValueError("Source and output must be separate directories")
    if not (source / "COMPLETE.txt").exists():
        raise ValueError("Incomplete source dataset")
    inputs = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    names = OmegaConf.load(source / "data.yaml").names
    rows, excluded, seen = [], [], set()
    for row in inputs:
        image, label = local_path(source, row["image"]), local_path(source, row["label"])
        if (
            checksum(image) != row["source_image_sha256"]
            or checksum(label) != row["cleaned_label_sha256"]
        ):
            raise ValueError(f"Source changed since cleaning: {image}")
        count = label_count(label, len(names))
        if count != row["kept_boxes"]:
            raise ValueError(f"Source box count changed: {label}")
        if cfg.exclude_empty and count == 0:
            excluded.append(dict(row, reason="empty_label_not_confirmed_as_background"))
            continue
        digest = row["source_image_sha256"]
        if digest in seen:
            raise ValueError(f"Duplicate image: {image}")
        seen.add(digest)
        video = Path(row["video"]).stem
        if Path(video).name != video or video in {"", ".", ".."}:
            raise ValueError("Invalid video ID")
        rows.append(
            dict(
                image=f"yolo/images/{video}/{image.name}",
                label=f"yolo/labels/{video}/{label.name}",
                video=row["video"],
                fold=row["fold"],
                frame_index=row["frame_index"],
                timestamp_ms=row["timestamp_ms"],
                boxes=count,
                source="sam3_assisted_annotation_cleaned",
                sha256=digest,
                label_sha256=row["cleaned_label_sha256"],
                original_image=row["image"],
                original_label=row["label"],
            )
        )
    folds = make_folds(rows, cfg.expected_folds)
    output.mkdir(parents=True)
    OmegaConf.save(cfg, output / "build_config.yaml")
    OmegaConf.save(OmegaConf.create(dict(names=list(names))), output / "classes.yaml")
    for row in rows:
        for key in ("image", "label"):
            target = local_path(output, row[key])
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(local_path(source, row[f"original_{key}"]), target)
    write_json(output / "yolo/manifest.json", rows)
    write_json(output / "yolo/folds.json", folds)
    write_json(output / "excluded.json", excluded)
    for fold in folds:
        folder = output / "folds" / f"fold_{fold['fold']}"
        folder.mkdir(parents=True)
        for split in ("train", "val"):
            (folder / f"{split}.txt").write_text(
                "\n".join(f"./../../{path}" for path in fold[split]) + "\n", encoding="utf-8"
            )
        OmegaConf.save(
            OmegaConf.create(
                dict(train="train.txt", val="val.txt", nc=len(names), names=list(names))
            ),
            folder / "data.yaml",
        )
    shutil.copy2(source / "report.json", output / "cleaning_report.json")
    for filename in ("source_README.dataset.txt", "source_README.roboflow.txt"):
        shutil.copy2(source / filename, output / filename)
    summary = validate_dataset(output, cfg.expected_folds)
    summary.update(
        excluded_empty=len(excluded),
        source_images=len(inputs),
        source_manifest_sha256=checksum(source / "manifest.json"),
    )
    write_json(output / "validation.json", summary)
    (output / "README.txt").write_text(
        "Five video groups; images and labels stored once under yolo/.\n"
        "folds/fold_N/data.yaml and train.txt/val.txt use relocatable paths.\n"
        "yolo/manifest.json works with lenta_price_tag_scanner.train_oof.\n"
        "Unconfirmed empty frames are excluded; see excluded.json.\n"
        "Training is not executed by the dataset builder.\n",
        encoding="utf-8",
    )
    (output / "COMPLETE.txt").write_text("Fold build and validation complete.\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
