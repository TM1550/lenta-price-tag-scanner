"""Inference module for trained model."""

import logging
from pathlib import Path
from typing import Union

from omegaconf import DictConfig
from ultralytics import YOLO

logger = logging.getLogger(__name__)


def load_model(cfg: DictConfig) -> YOLO:
    """Load trained model.

    Args:
        cfg: Configuration dictionary.

    Returns:
        Loaded YOLO model.
    """
    models_dir = Path(cfg.paths.models_dir)

    best_model = models_dir / "best.pt"
    if best_model.exists():
        logger.info(f"Loading best model from {best_model}")
        return YOLO(str(best_model))

    onnx_model = models_dir / "exported" / "best.onnx"
    if onnx_model.exists():
        logger.info(f"Loading ONNX model from {onnx_model}")
        return YOLO(str(onnx_model))

    logger.warning("No trained model found, using base model")
    return YOLO(cfg.model.model_name)


def run_inference(cfg: DictConfig, source: Union[str, Path, int] = 0) -> None:
    """Run inference on images or video.

    Args:
        cfg: Configuration dictionary.
        source: Source for inference (image path, video path, or camera index).
    """
    model = load_model(cfg)

    logger.info(f"Running inference on source: {source}")

    results = model.predict(
        source=source,
        imgsz=cfg.data.img_size,
        conf=cfg.train.validation.conf,
        iou=cfg.train.validation.iou,
        device=cfg.train.validation.device,
        save=True,
        save_txt=cfg.train.validation.save_txt,
        save_conf=cfg.train.validation.save_conf,
    )

    for result in results:
        logger.info(f"Detected {len(result.boxes)} objects")
        if len(result.boxes) > 0:
            logger.info(f"Boxes: {result.boxes.xyxy}")
            logger.info(f"Confidences: {result.boxes.conf}")
