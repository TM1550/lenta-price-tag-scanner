"""Package a trained YOLO detector as an MLflow PyFunc model."""

import base64
from importlib.metadata import version
from pathlib import Path

import hydra

from .build_datasets import checksum, resolve, write_json
from .data_loader import download_data_with_dvc


@hydra.main(version_base=None, config_path="../configs", config_name="serving_yolo")
def main(cfg):
    import mlflow
    import pandas as pd

    from .serving_yolo_model import PriceTagDetector

    download_data_with_dvc(cfg.dvc_targets, cfg.pull_dvc)
    checkpoint = resolve(cfg.checkpoint)
    example = pd.DataFrame(
        dict(image_base64=[base64.b64encode(resolve(cfg.example).read_bytes()).decode("ascii")])
    )
    output = resolve(cfg.output)
    if output.exists():
        raise FileExistsError(output)
    signature = mlflow.models.infer_signature(
        example, pd.DataFrame(dict(detections_json=["[]"], num_detections=[0]))
    )
    mlflow.pyfunc.save_model(
        path=str(output),
        python_model=PriceTagDetector(cfg.imgsz, cfg.conf, cfg.iou),
        artifacts=dict(checkpoint=str(checkpoint)),
        signature=signature,
        input_example=example,
        code_paths=[str(Path(__file__).resolve().parent)],
        pip_requirements=[
            f"{name}=={version(name)}"
            for name in ["mlflow", "ultralytics", "torch", "torchvision", "Pillow", "pandas"]
        ],
    )
    predicted = mlflow.pyfunc.load_model(str(output)).predict(example)
    write_json(
        output / "roundtrip.json",
        dict(
            checkpoint_sha256=checksum(checkpoint), predictions=predicted.to_dict(orient="records")
        ),
    )
    write_json(
        output / "request.json",
        dict(dataframe_split=dict(columns=["image_base64"], data=example.values.tolist())),
    )


if __name__ == "__main__":
    main()
