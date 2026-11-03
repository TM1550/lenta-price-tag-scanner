"""Package a trained classifier as an MLflow model and verify round-trip prediction."""

import base64
from importlib.metadata import version
from pathlib import Path

import hydra

from .build_datasets import resolve, write_json
from .data_loader import download_data_with_dvc


@hydra.main(version_base=None, config_path="../configs", config_name="serving")
def main(cfg):
    import mlflow
    import pandas as pd

    from .serving_model import CropClassifier

    download_data_with_dvc(cfg.dvc_targets, cfg.pull_dvc)
    example = pd.DataFrame(
        {"image_base64": [base64.b64encode(resolve(cfg.example).read_bytes()).decode()]}
    )
    signature = mlflow.models.infer_signature(example, pd.DataFrame({"probability": [0.5]}))
    output = resolve(cfg.output)
    if output.exists():
        raise FileExistsError(output)
    mlflow.pyfunc.save_model(
        path=str(output),
        python_model=CropClassifier(),
        artifacts={"checkpoint": str(resolve(cfg.checkpoint))},
        signature=signature,
        input_example=example,
        code_paths=[str(Path(__file__).resolve().parent)],
        pip_requirements=[
            f"{name}=={version(name)}"
            for name in ["mlflow", "torch", "torchvision", "timm", "Pillow", "pandas"]
        ],
    )
    predicted = mlflow.pyfunc.load_model(str(output)).predict(example)
    probability = float(predicted["probability"].iloc[0])
    if not 0 <= probability <= 1:
        raise ValueError("Invalid serving probability")
    write_json(output / "roundtrip.json", {"probability": probability})
    write_json(
        output / "request.json",
        {"dataframe_split": {"columns": ["image_base64"], "data": example.values.tolist()}},
    )


if __name__ == "__main__":
    main()
