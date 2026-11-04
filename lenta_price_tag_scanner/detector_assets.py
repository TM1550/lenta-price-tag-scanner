"""Resolve explicit, held-out-video or documented default detector checkpoints."""

import json
from pathlib import Path

from .build_datasets import checksum, resolve


def detector_for_video(cfg, video):
    name = Path(video).name
    checkpoint = cfg.get("yolo_weights")
    entry, source = {}, "explicit"
    if not checkpoint:
        manifest = cfg.get("yolo_model_map")
        if manifest:
            manifest = resolve(manifest)
            models = json.loads(manifest.read_text(encoding="utf-8"))
            entry = models.get(name, {})
            if entry:
                checkpoint = (manifest.parent / entry["model"]).resolve()
                if not checkpoint.is_relative_to(manifest.parent.resolve()):
                    raise ValueError("Detector checkpoint escapes model-map directory")
                if name in {Path(value).name for value in entry.get("train_videos", [])}:
                    raise ValueError("Held-out detector map contains training-video leakage")
                source = "held_out_video_map"
        if not checkpoint:
            checkpoint = cfg.get("yolo_default_weights")
            source = "deployment_default"
    if not checkpoint:
        raise ValueError(f"Specify yolo_weights for video {name}")
    path = resolve(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(path)
    sha256 = checksum(path)
    if entry and sha256 != entry["sha256"]:
        raise ValueError(f"Detector checkpoint checksum mismatch: {name}")
    return dict(checkpoint=str(path), sha256=sha256, source=source, fold=entry.get("fold"))


def classifier_selection(rank, probability, mode, threshold, weight):
    """Rank mode retains low-scoring crops; only explicit filter mode rejects them."""
    if mode not in {"off", "rank", "filter"}:
        raise ValueError("efficient_mode must be off, rank or filter")
    if not 0 <= weight <= 1 or not 0 <= threshold <= 1:
        raise ValueError("Classifier thresholds/weights must be within [0, 1]")
    if probability is None or mode == "off":
        return rank, False
    if not 0 <= probability <= 1:
        raise ValueError("Invalid classifier probability")
    if mode == "filter":
        return rank, probability < threshold
    return rank * (1 - weight + weight * probability), False
