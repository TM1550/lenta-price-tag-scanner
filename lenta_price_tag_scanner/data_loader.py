"""Data loading and management using DVC."""

import logging
from pathlib import Path

import dvc.api

logger = logging.getLogger(__name__)


def download_data_with_dvc(data_dir: Path) -> None:
    """Download data using DVC.

    Args:
        data_dir: Directory where data should be stored.
    """
    logger.info(f"Downloading data to {data_dir}")

    data_dir.mkdir(parents=True, exist_ok=True)

    try:
        dvc.api.open_repo(".", subrepos=True)
        logger.info("DVC repository found, data should be available")
    except Exception as e:
        logger.warning(f"DVC repo not configured: {e}")
        logger.info("Make sure to run 'dvc pull' to get the data")

    logger.info(f"Data directory: {data_dir}")


def get_data_yaml_path(data_dir: Path) -> Path:
    """Get path to data.yaml file.

    Args:
        data_dir: Data directory path.

    Returns:
        Path to data.yaml file.
    """
    data_yaml = data_dir / "data.yaml"

    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found at {data_yaml}")

    return data_yaml
