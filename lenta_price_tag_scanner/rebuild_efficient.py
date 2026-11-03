"""Build a new classifier dataset exclusively from manually reviewed labels."""

import csv
import json
import shutil
from pathlib import Path

import hydra
from PIL import Image

from .build_datasets import resolve, write_json


@hydra.main(version_base=None, config_path="../configs", config_name="rebuild_efficient")
def main(cfg):
    source = resolve(cfg.source)
    output = resolve(cfg.output)
    if output.exists():
        raise FileExistsError(output)
    original = {
        row["crop"]: row
        for row in (
            json.loads(line)
            for line in (source / "source_manifest.jsonl").read_text(encoding="utf-8").splitlines()
        )
    }
    selected, seen = [], set()
    with (source / "labels.csv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["reviewed"].strip().lower() != "true":
                continue
            if row["label"].strip() not in ("0", "1"):
                raise ValueError(f"Reviewed row needs label 0 or 1: {row['image']}")
            if row["image"] not in original or row["image"] in seen:
                raise ValueError(f"Unknown or duplicate image: {row['image']}")
            seen.add(row["image"])
            metadata = original[row["image"]]
            path = (source / row["image"]).resolve()
            path.relative_to(source.resolve())
            with Image.open(path) as image:
                image.verify()
            selected.append((path, metadata, int(row["label"].strip())))
    if {item[1]["fold"] for item in selected} != set(range(5)):
        raise ValueError("Review examples from all five videos before rebuilding five folds")
    for fold in range(5):
        if {label for _, metadata, label in selected if metadata["fold"] != fold} != {0, 1}:
            raise ValueError(f"Training partition {fold} must contain both classes")
    rows = []
    for path, metadata, label in selected:
        video = Path(metadata["video"]).name
        relative = Path("efficient/images") / str(label) / Path(video).stem / path.name
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        rows.append(
            dict(
                image=relative.as_posix(),
                label=label,
                video=video,
                fold=metadata["fold"],
                source="human_review",
                reviewed=True,
            )
        )
    write_json(output / "efficient/manifest.json", rows)
    write_json(
        output / "efficient/folds.json",
        [
            dict(
                fold=fold,
                train=[row["image"] for row in rows if row["fold"] != fold],
                val=[row["image"] for row in rows if row["fold"] == fold],
            )
            for fold in range(5)
        ],
    )
    print(f"Saved {len(rows)} manually reviewed images: {output}")


if __name__ == "__main__":
    main()
