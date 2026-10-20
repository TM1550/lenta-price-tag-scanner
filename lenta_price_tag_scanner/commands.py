"""Main entry point for CLI commands."""

import logging

import hydra
from omegaconf import DictConfig

from .config import setup_config
from .data_loader import download_data_with_dvc
from .trainer import YOLOv8Trainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def train(cfg: DictConfig) -> None:
    """Train YOLOv8 model.

    Args:
        cfg: Configuration dictionary.
    """
    logger.info("Starting training pipeline...")

    cfg = setup_config(cfg)

    from pathlib import Path

    data_dir = Path(cfg.data.data_path)
    download_data_with_dvc(data_dir)

    trainer = YOLOv8Trainer(cfg)
    trainer.train()

    logger.info("Training pipeline completed!")


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def infer(cfg: DictConfig) -> None:
    """Run inference with trained model.

    Args:
        cfg: Configuration dictionary.
    """
    logger.info("Starting inference...")

    cfg = setup_config(cfg)

    from .inference import run_inference

    run_inference(cfg)

    logger.info("Inference completed!")


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def export(cfg: DictConfig) -> None:
    """Export model to ONNX and TensorRT.

    Args:
        cfg: Configuration dictionary.
    """
    logger.info("Starting model export...")

    cfg = setup_config(cfg)

    from .export_model import export_model

    export_model(cfg)

    logger.info("Model export completed!")


def main() -> None:
    """Main entry point."""
    import fire

    commands = {
        "train": train,
        "infer": infer,
        "export": export,
    }

    fire.Fire(commands)


if __name__ == "__main__":
    main()
