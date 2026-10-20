"""Training module for YOLOv8 model."""

import logging
from pathlib import Path
from typing import Any, Dict

import git
import mlflow
from omegaconf import DictConfig
from ultralytics import YOLO

logger = logging.getLogger(__name__)


def get_git_commit_hash() -> str:
    """Get current git commit hash."""
    try:
        repo = git.Repo(search_parent_directories=True)
        return repo.head.object.hexsha
    except Exception as e:
        logger.warning(f"Could not get git commit hash: {e}")
        return "unknown"


class YOLOv8Trainer:
    """Trainer for YOLOv8 models with MLflow logging."""

    def __init__(self, cfg: DictConfig):
        """Initialize trainer.

        Args:
            cfg: Configuration dictionary.
        """
        self.cfg = cfg
        self.model = None
        self.mlflow_run = None

    def setup_mlflow(self) -> None:
        """Setup MLflow tracking."""
        mlflow_uri = self.cfg.project.mlflow_uri
        experiment_name = self.cfg.project.experiment_name

        logger.info(f"Setting up MLflow at {mlflow_uri}")
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment(experiment_name)

    def load_model(self) -> None:
        """Load YOLOv8 model."""
        model_name = self.cfg.model.model_name
        logger.info(f"Loading model: {model_name}")

        self.model = YOLO(model_name)
        logger.info("Model loaded successfully")

    def train(self) -> None:
        """Train the model."""
        if self.model is None:
            self.load_model()

        self.setup_mlflow()

        data_yaml = Path(self.cfg.data.data_yaml)
        logger.info(f"Using data config: {data_yaml}")

        train_args = {
            "data": str(data_yaml),
            "epochs": self.cfg.train.epochs,
            "patience": self.cfg.train.patience,
            "batch": self.cfg.train.batch,
            "imgsz": self.cfg.train.imgsz,
            "device": self.cfg.train.device,
            "workers": self.cfg.train.workers,
            "optimizer": self.cfg.train.optimizer,
            "lr0": self.cfg.train.lr0,
            "lrf": self.cfg.train.lrf,
            "momentum": self.cfg.train.momentum,
            "weight_decay": self.cfg.train.weight_decay,
            "warmup_epochs": self.cfg.train.warmup_epochs,
            "warmup_momentum": self.cfg.train.warmup_momentum,
            "warmup_bias_lr": self.cfg.train.warmup_bias_lr,
            "box": self.cfg.train.box,
            "cls": self.cfg.train.cls,
            "dfl": self.cfg.train.dfl,
            "overlap_mask": self.cfg.train.overlap_mask,
            "mask_ratio": self.cfg.train.mask_ratio,
            "hsv_h": self.cfg.train.augmentation.hsv_h,
            "hsv_s": self.cfg.train.augmentation.hsv_s,
            "hsv_v": self.cfg.train.augmentation.hsv_v,
            "degrees": self.cfg.train.augmentation.degrees,
            "translate": self.cfg.train.augmentation.translate,
            "scale": self.cfg.train.augmentation.scale,
            "shear": self.cfg.train.augmentation.shear,
            "perspective": self.cfg.train.augmentation.perspective,
            "flipud": self.cfg.train.augmentation.flipud,
            "fliplr": self.cfg.train.augmentation.fliplr,
            "mosaic": self.cfg.train.augmentation.mosaic,
            "mixup": self.cfg.train.augmentation.mixup,
            "copy_paste": self.cfg.train.augmentation.copy_paste,
            "save": self.cfg.train.validation.save,
            "save_json": self.cfg.train.validation.save_json,
            "save_hybrid": self.cfg.train.validation.save_hybrid,
            "save_conf": self.cfg.train.validation.save_conf,
            "save_txt": self.cfg.train.validation.save_txt,
            "iou": self.cfg.train.validation.iou,
            "conf": self.cfg.train.validation.conf,
            "max_det": self.cfg.train.validation.max_det,
            "half": self.cfg.train.validation.half,
            "verbose": True,
        }

        git_hash = get_git_commit_hash()

        with mlflow.start_run() as run:
            self.mlflow_run = run

            self._log_params(train_args)
            mlflow.log_param("git_commit", git_hash)

            self.model.add_callback("on_train_start", self._on_train_start)
            self.model.add_callback("on_epoch_end", self._on_epoch_end)
            self.model.add_callback("on_train_end", self._on_train_end)

            logger.info("Starting training...")
            results = self.model.train(**train_args)

            logger.info("Training completed!")

            self._log_final_metrics(results)
            self._save_model_artifact(results)

    def _log_params(self, train_args: Dict[str, Any]) -> None:
        """Log training parameters to MLflow."""
        mlflow.log_params(train_args)
        mlflow.log_params(
            {
                "model_name": self.cfg.model.model_name,
                "pretrained": self.cfg.model.pretrained,
                "project_name": self.cfg.project.name,
            }
        )

    def _on_train_start(self, trainer) -> None:
        """Callback for training start."""
        logger.info("Training started")

    def _on_epoch_end(self, trainer) -> None:
        """Callback for epoch end - log metrics."""
        if self.mlflow_run is None:
            return

        epoch = trainer.epoch + 1

        if hasattr(trainer, "metrics"):
            metrics = trainer.metrics

            if "train/box_loss" in metrics:
                mlflow.log_metric("train/box_loss", metrics["train/box_loss"], step=epoch)
            if "train/cls_loss" in metrics:
                mlflow.log_metric("train/cls_loss", metrics["train/cls_loss"], step=epoch)
            if "train/dfl_loss" in metrics:
                mlflow.log_metric("train/dfl_loss", metrics["train/dfl_loss"], step=epoch)

            if "metrics/precision(B)" in metrics:
                mlflow.log_metric("val/precision", metrics["metrics/precision(B)"], step=epoch)
            if "metrics/recall(B)" in metrics:
                mlflow.log_metric("val/recall", metrics["metrics/recall(B)"], step=epoch)
            if "metrics/mAP50(B)" in metrics:
                mlflow.log_metric("val/mAP50", metrics["metrics/mAP50(B)"], step=epoch)
            if "metrics/mAP50-95(B)" in metrics:
                mlflow.log_metric("val/mAP50-95", metrics["metrics/mAP50-95(B)"], step=epoch)

    def _on_train_end(self, trainer) -> None:
        """Callback for training end."""
        logger.info("Training ended")

    def _log_final_metrics(self, results) -> None:
        """Log final training metrics."""
        if hasattr(results, "results_dict"):
            metrics = results.results_dict

            mlflow.log_metric("final/box_loss", metrics.get("train/box_loss", 0))
            mlflow.log_metric("final/precision", metrics.get("metrics/precision(B)", 0))
            mlflow.log_metric("final/recall", metrics.get("metrics/recall(B)", 0))
            mlflow.log_metric("final/mAP50", metrics.get("metrics/mAP50(B)", 0))
            mlflow.log_metric("final/mAP50-95", metrics.get("metrics/mAP50-95(B)", 0))

    def _save_model_artifact(self, results) -> None:
        """Save model as MLflow artifact."""
        if hasattr(results, "best"):
            best_model_path = results.best
            mlflow.log_artifact(str(best_model_path), "models")
            logger.info(f"Logged best model: {best_model_path}")

        if hasattr(results, "last"):
            last_model_path = results.last
            mlflow.log_artifact(str(last_model_path), "models")
            logger.info(f"Logged last model: {last_model_path}")

        if hasattr(results, "save_dir"):
            save_dir = Path(results.save_dir)
            results_plot = save_dir / "results.png"
            if results_plot.exists():
                mlflow.log_artifact(str(results_plot), "plots")
                logger.info(f"Logged results plot: {results_plot}")
