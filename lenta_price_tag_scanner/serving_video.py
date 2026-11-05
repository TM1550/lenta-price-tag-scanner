"""Serve MP4 videos through the same video-to-CSV pipeline as batch inference."""

import base64
import tempfile
from pathlib import Path

import click
import mlflow.pyfunc
import pandas as pd
from hydra import compose, initialize_config_module
from omegaconf import OmegaConf

from .data_loader import download_data_with_dvc
from .video_to_csv import main as infer


class VideoScanner(mlflow.pyfunc.PythonModel):
    def __init__(self, config):
        self.config = config

    def predict(self, context, model_input, params=None):
        results = []
        for row in model_input.itertuples(index=False):
            name = row.filename
            if Path(name).name != name or "/" in name or "\\" in name or not name.endswith(".mp4"):
                raise ValueError("Expected an MP4 basename")
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / name).write_bytes(base64.b64decode(row.video_base64, validate=True))
                config = OmegaConf.create(self.config)
                config.videos = str(root)
                config.output = str(root / "result")
                config.catalogue_db = str(root / "catalogue.sqlite")
                infer.__wrapped__(config)
                results.append((root / "result/submission.csv").read_text(encoding="utf-8-sig"))
        return pd.DataFrame({"csv": results})


@click.command()
@click.option("--output", required=True, type=click.Path(path_type=Path))
@click.argument("overrides", nargs=-1)
def main(output, overrides):
    """Package the video pipeline with references to the deployed DVC assets."""
    with initialize_config_module(config_module="configs", version_base=None):
        config = compose(config_name="video_to_csv", overrides=list(overrides))
    download_data_with_dvc(config.dvc_targets, config.pull_dvc)
    config.pull_dvc = False
    config.resume_tracks = None
    config.pattern = "*.mp4"
    signature = mlflow.models.infer_signature(
        pd.DataFrame({"filename": ["video.mp4"], "video_base64": [""]}),
        pd.DataFrame({"csv": [""]}),
    )
    mlflow.pyfunc.save_model(
        path=str(output),
        python_model=VideoScanner(OmegaConf.to_container(config, resolve=True)),
        signature=signature,
        pip_requirements=["lenta-price-tag-scanner==0.1.0"],
    )


if __name__ == "__main__":
    main()
