"""Validate image integrity, fold coverage and absence of video leakage."""

import json
from collections import Counter

import hydra
from PIL import Image

from .build_datasets import checksum, resolve, write_json


@hydra.main(version_base=None, config_path="../configs", config_name="train_oof")
def main(cfg):
    root = resolve(cfg.dataset)
    report = {}
    for kind, expected in [("yolo", 4), ("efficient", 5)]:
        rows = json.loads((root / kind / "manifest.json").read_text())
        folds = json.loads((root / kind / "folds.json").read_text())
        assert len(folds) == expected
        by_image = {row["image"]: row for row in rows}
        assert len(by_image) == len(rows)
        seen_hashes = {}
        for row in rows:
            path = root / row["image"]
            with Image.open(path) as image:
                image.verify()
            digest = checksum(path)
            assert digest not in seen_hashes or seen_hashes[digest] == row["video"]
            seen_hashes[digest] = row["video"]
        coverage = Counter()
        for fold in folds:
            train, valid = set(fold["train"]), set(fold["val"])
            assert train and valid and train.isdisjoint(valid)
            assert train | valid == set(by_image)
            training_videos = {by_image[path]["video"] for path in train}
            valid_videos = {by_image[path]["video"] for path in valid}
            assert len(valid_videos) == 1 and training_videos.isdisjoint(valid_videos)
            coverage.update(valid)
        assert set(coverage.values()) == {1}
        report[kind] = dict(
            images=len(rows), folds=len(folds), videos=dict(Counter(row["video"] for row in rows))
        )
        if kind == "efficient":
            assert {row["label"] for row in rows} == {0, 1}
            report[kind]["labels"] = dict(Counter(row["label"] for row in rows))
    write_json(root / "validation.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
