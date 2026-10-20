"""Configuration module."""

from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def get_project_root() -> Path:
    """Get project root directory."""
    return Path(__file__).parent.parent


def setup_config(cfg: DictConfig) -> DictConfig:
    """Setup and validate configuration."""
    if "PROJECT_ROOT" not in cfg.paths:
        cfg.paths.root_dir = str(get_project_root())

    OmegaConf.resolve(cfg)

    return cfg
