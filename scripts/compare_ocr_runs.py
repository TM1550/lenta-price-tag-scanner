"""Summarize fixed-set OCR experiments and quality selection without hiding retention."""

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import fire
from omegaconf import OmegaConf
from PIL import Image

from lenta_price_tag_scanner.crop_quality import quality, rejection_reasons


def metrics(rows):
    return dict(
        images=len(rows),
        cer=sum(row["char_errors"] for row in rows) / max(1, sum(row["chars"] for row in rows)),
        wer=sum(row["word_errors"] for row in rows) / max(1, sum(row["words"] for row in rows)),
    )


def main(images, outputs, result, runs):
    args = SimpleNamespace(
        images=Path(images), outputs=Path(outputs), result=Path(result), runs=runs
    )
    cfg = OmegaConf.create(
        dict(min_side=40, min_sharpness=15, min_contrast=12, max_flat_fraction=0.98)
    )
    scores, keep, groups = {}, set(), defaultdict(list)
    for path in sorted(args.images.glob("*.png")):
        with Image.open(path) as image:
            values = quality(image)
        scores[path.name] = math.log1p(values["sharpness"]) * math.sqrt(values["min_side"])
        if not rejection_reasons(values, cfg):
            keep.add(path.name)
        groups[path.name.split("_frame")[0]].append(path.name)
    best = {max(names, key=lambda name: scores[name]) for names in groups.values()}
    first = {names[0] for names in groups.values()}
    report = dict(
        quality=OmegaConf.to_container(cfg),
        images=len(scores),
        retained=len(keep),
        tracks=len(groups),
        runs={},
        note="Development benchmark. Full data used to compare methods; no untouched test claimed.",
    )
    for run in args.runs:
        root = args.outputs / run
        rows = json.loads((root / "evaluation.json").read_text(encoding="utf-8"))
        if set(row["image"] for row in rows) != set(scores):
            raise ValueError(f"Experiment {run} does not cover the same complete image set")
        summary = json.loads((root / "summary.json").read_text())
        report["runs"][run] = dict(
            all=metrics(rows),
            quality_pass=metrics([row for row in rows if row["image"] in keep]),
            best_per_track=metrics([row for row in rows if row["image"] in best]),
            first_per_track=metrics([row for row in rows if row["image"] in first]),
            processing_seconds=summary["processing_seconds"],
            peak_gpu_bytes=summary["peak_gpu_bytes"],
            per_video=summary["per_video"],
        )
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(report, indent=2), encoding="utf-8")
    with args.result.with_suffix(".csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["run", "images", "cer", "wer", "seconds", "gpu_gib"]
        )
        writer.writeheader()
        for name, result in report["runs"].items():
            writer.writerow(
                dict(
                    run=name,
                    **result["all"],
                    seconds=result["processing_seconds"],
                    gpu_gib=result["peak_gpu_bytes"] / 1024**3,
                )
            )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ordered = sorted(report["runs"], key=lambda name: report["runs"][name]["all"]["cer"])
    fig, axis = plt.subplots(figsize=(10, 5))
    bars = axis.barh(ordered, [report["runs"][name]["all"]["cer"] for name in ordered])
    axis.invert_yaxis()
    axis.set_xlim(0, 1)
    axis.set_xlabel("Character error rate (lower is better), same 294 crops")
    axis.bar_label(bars, fmt="%.3f", padding=4)
    fig.tight_layout()
    fig.savefig(args.result.with_suffix(".png"), dpi=150)
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    fire.Fire(main)
