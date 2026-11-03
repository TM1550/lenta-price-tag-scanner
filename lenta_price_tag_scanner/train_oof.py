"""Five video folds; metrics measure pseudo-label agreement, not human accuracy."""

import json
import re

import hydra
from omegaconf import OmegaConf

from .build_datasets import checksum, resolve, write_json
from .data_loader import code_version, download_data_with_dvc


def prepare(cfg, kind, fold):
    root = resolve(cfg.dataset)
    rows = json.loads((root / kind / "manifest.json").read_text())
    train = [row for row in rows if row["fold"] != fold]
    valid = [row for row in rows if row["fold"] == fold]
    assert {row["video"] for row in train}.isdisjoint({row["video"] for row in valid})
    if cfg.smoke:
        train, valid = train[:8], valid[:8]
    if not train or not valid:
        raise ValueError("Empty fold")
    return root, train, valid


def efficient(cfg, fold, output):
    import lightning as lightning
    import mlflow
    import timm
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms

    root, training, validation = prepare(cfg, "efficient", fold)
    transform = transforms.Compose(
        [
            transforms.Resize((cfg.oof.size, cfg.oof.size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    class Crops(Dataset):
        def __init__(self, rows):
            self.rows = rows

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index):
            row = self.rows[index]
            with Image.open(root / row["image"]) as image:
                pixels = transform(image.convert("RGB"))
            return pixels, torch.tensor(float(row["label"]))

    class Classifier(lightning.LightningModule):
        def __init__(self):
            super().__init__()
            self.model = timm.create_model(
                cfg.oof.architecture, pretrained=cfg.oof.pretrained, num_classes=1
            )

        def forward(self, pixels):
            return self.model(pixels).flatten()

        def training_step(self, batch, batch_index):
            pixels, labels = batch
            loss = torch.nn.functional.binary_cross_entropy_with_logits(self(pixels), labels)
            self.log("train_loss", loss, on_step=False, on_epoch=True, batch_size=len(labels))
            return loss

        def configure_optimizers(self):
            return torch.optim.AdamW(
                self.parameters(), lr=cfg.oof.lr, weight_decay=cfg.oof.weight_decay
            )

    class Metrics(lightning.Callback):
        def on_train_epoch_end(self, trainer, module):
            mlflow.log_metric(
                "train_loss",
                float(trainer.callback_metrics["train_loss"]),
                step=trainer.current_epoch,
            )

    lightning.seed_everything(cfg.seed + fold, workers=True)
    model = Classifier()
    loader = DataLoader(
        Crops(training), batch_size=cfg.oof.batch, shuffle=True, num_workers=cfg.oof.workers
    )
    trainer = lightning.Trainer(
        max_epochs=1 if cfg.smoke else cfg.oof.epochs,
        accelerator=cfg.oof.accelerator,
        devices=1,
        logger=lightning.pytorch.loggers.CSVLogger(str(output), name="history"),
        callbacks=[Metrics()],
        enable_checkpointing=False,
        deterministic=True,
        default_root_dir=str(output),
        enable_progress_bar=False,
    )
    # Held-out video is evaluated only after fixed-epoch training: no checkpoint selection.
    trainer.fit(model, loader)
    model.cpu().eval()
    torch.save(
        dict(
            state_dict=model.model.state_dict(),
            architecture=cfg.oof.architecture,
            size=cfg.oof.size,
            color="RGB",
            positive_class=1,
        ),
        output / "model.pt",
    )
    probabilities, targets = [], []
    with torch.inference_mode():
        for pixels, labels in DataLoader(Crops(validation), batch_size=cfg.oof.batch):
            probabilities.extend(torch.sigmoid(model(pixels)).tolist())
            targets.extend(labels.tolist())
    scores = torch.tensor(probabilities)
    truth = torch.tensor(targets).bool()
    predicted = scores >= cfg.oof.threshold
    true_positive = int((predicted & truth).sum())
    false_positive = int((predicted & ~truth).sum())
    false_negative = int((~predicted & truth).sum())
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    metrics = dict(
        precision=precision,
        recall=recall,
        f1=2 * precision * recall / max(1e-12, precision + recall),
        accuracy=float((predicted == truth).float().mean()),
        val_loss=float(torch.nn.functional.binary_cross_entropy(scores, torch.tensor(targets))),
    )
    predictions = [
        dict(**row, probability=probability) for row, probability in zip(validation, probabilities)
    ]
    write_json(output / "oof_predictions.json", predictions)
    return metrics


def yolo(cfg, fold, output):
    import mlflow
    from ultralytics import YOLO

    root, training, validation = prepare(cfg, "yolo", fold)
    for name, rows in [("train", training), ("val", validation)]:
        (output / f"{name}.txt").write_text(
            "\n".join(str(root / row["image"]) for row in rows) + "\n"
        )
    data = dict(
        path=str(root),
        train=str(output / "train.txt"),
        val=str(output / "val.txt"),
        names={0: "price_tag"},
    )
    OmegaConf.save(OmegaConf.create(data), output / "data.yaml")
    model = YOLO(cfg.oof.architecture)

    def log_epoch(trainer):
        values = trainer.label_loss_items(trainer.tloss, prefix="train")
        mlflow.log_metrics({key: float(value) for key, value in values.items()}, step=trainer.epoch)

    model.add_callback("on_train_epoch_end", log_epoch)
    model.train(
        data=str(output / "data.yaml"),
        epochs=1 if cfg.smoke else cfg.oof.epochs,
        imgsz=cfg.oof.size,
        batch=cfg.oof.batch,
        workers=cfg.oof.workers,
        device=cfg.oof.device,
        lr0=cfg.oof.lr,
        seed=cfg.seed + fold,
        project=str(output),
        name="training",
        exist_ok=False,
        patience=0,
        pretrained=False,
        val=False,
        plots=True,
    )
    # Use last, never best selected on the outer validation video.
    trained = YOLO(str(output / "training/weights/last.pt"))
    result = trained.val(
        data=str(output / "data.yaml"),
        imgsz=cfg.oof.size,
        device=cfg.oof.device,
        project=str(output),
        name="evaluation",
    )
    predictions = []
    for row in validation:
        result_image = trained.predict(
            str(root / row["image"]), imgsz=cfg.oof.size, device=cfg.oof.device, verbose=False
        )[0]
        predictions.append(dict(**row, detections=result_image.boxes.data.cpu().tolist()))
    write_json(output / "oof_predictions.json", predictions)
    return {key: float(value) for key, value in result.results_dict.items()}


def plots(history, output):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    destination = output / "plots"
    destination.mkdir()
    for metric in history[0]["metrics"]:
        figure, axes = plt.subplots()
        axes.plot(
            [row["fold"] for row in history],
            [row["metrics"][metric] for row in history],
            marker="o",
        )
        axes.set(
            xlabel="Held-out video fold",
            ylabel=metric,
            title="OOF metrics (EfficientNet uses teacher labels)",
        )
        figure.tight_layout()
        figure.savefig(destination / (metric.replace("/", "_") + ".png"))
        plt.close(figure)


@hydra.main(version_base=None, config_path="../configs", config_name="train_oof")
def main(cfg):
    import mlflow

    download_data_with_dvc(cfg.dvc_targets, cfg.pull_dvc)
    rows = json.loads((resolve(cfg.dataset) / cfg.oof.kind / "manifest.json").read_text())
    available = sorted({row["fold"] for row in rows})
    selected = available if cfg.folds is None else list(cfg.folds)
    if len(set(selected)) != len(selected) or not set(selected).issubset(available):
        raise ValueError(f"Choose unique folds from {available}")
    output = resolve(cfg.output)
    output.mkdir(parents=True, exist_ok=False)
    OmegaConf.save(cfg, output / "config.yaml")
    mlflow.set_tracking_uri(cfg.mlflow_uri)
    mlflow.set_experiment(cfg.experiment)
    history = []
    for fold in selected:
        fold_output = output / f"fold_{fold}"
        fold_output.mkdir()
        with mlflow.start_run(run_name=f"{cfg.oof.kind}_fold_{fold}"):
            mlflow.log_params(
                dict(
                    fold=fold,
                    seed=cfg.seed + fold,
                    **code_version(),
                    pseudo_labels=any(row.get("source") == "teacher_pseudo" for row in rows),
                    smoke=cfg.smoke,
                    manifest_sha256=checksum(resolve(cfg.dataset) / cfg.oof.kind / "manifest.json"),
                    **OmegaConf.to_container(cfg.oof),
                )
            )
            mlflow.log_artifact(str(output / "config.yaml"))
            metrics = (efficient if cfg.oof.kind == "efficient" else yolo)(cfg, fold, fold_output)
            mlflow.log_metrics(
                {re.sub(r"[^a-zA-Z0-9_./ -]", "_", key): value for key, value in metrics.items()}
            )
            mlflow.log_artifact(str(fold_output / "oof_predictions.json"))
            history.append(dict(fold=fold, metrics=metrics))
            write_json(output / "metrics.json", history)
    plots(history, output)
    with mlflow.start_run(run_name=f"{cfg.oof.kind}_oof_summary"):
        mlflow.log_params(code_version())
        mlflow.log_param("smoke", cfg.smoke)
        for row in history:
            mlflow.log_metrics(
                {
                    re.sub(r"[^a-zA-Z0-9_./ -]", "_", key): value
                    for key, value in row["metrics"].items()
                },
                step=row["fold"],
            )
        mlflow.log_artifacts(str(output / "plots"), artifact_path="plots")
        mlflow.log_artifact(str(output / "config.yaml"))


if __name__ == "__main__":
    main()
