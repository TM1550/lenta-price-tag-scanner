"""Model export to ONNX and TensorRT."""

import logging
from pathlib import Path

from omegaconf import DictConfig
from ultralytics import YOLO

logger = logging.getLogger(__name__)


def export_model(cfg: DictConfig) -> None:
    """Export model to ONNX and TensorRT formats.

    Args:
        cfg: Configuration dictionary.
    """
    # Load trained model (use best.pt if available)
    models_dir = Path(cfg.paths.models_dir)
    best_model = models_dir / "best.pt"

    if not best_model.exists():
        logger.warning(f"Best model not found at {best_model}, using yolov8n.pt")
        model = YOLO(cfg.model.model_name)
    else:
        logger.info(f"Loading best model from {best_model}")
        model = YOLO(str(best_model))

    # Create export directory
    export_dir = models_dir / "exported"
    export_dir.mkdir(parents=True, exist_ok=True)

    # Export to ONNX
    logger.info("Exporting to ONNX...")
    onnx_path = model.export(
        format="onnx",
        opset=12,
        simplify=True,
        dynamic=False,
    )
    logger.info(f"ONNX model saved to: {onnx_path}")

    # Export to TensorRT (if available)
    try:
        logger.info("Exporting to TensorRT...")
        engine_path = model.export(
            format="engine",
            device=cfg.train.device,
            half=True,
        )
        logger.info(f"TensorRT engine saved to: {engine_path}")
    except Exception as e:
        logger.warning(f"Could not export to TensorRT: {e}")
        logger.info("Make sure TensorRT is installed and GPU is available")
