"""One CLI; Click dispatches Hydra overrides without importing training dependencies."""

import importlib

import click
from hydra import compose, initialize_config_module


@click.command(context_settings={"ignore_unknown_options": True})
@click.argument(
    "command",
    type=click.Choice(
        [
            "train",
            "train-yolo",
            "infer",
            "infer-cpu",
            "track-eval",
            "ocr",
            "export",
            "export-yolo",
            "download",
            "serve-package",
            "serve-yolo-package",
        ]
    ),
)
@click.argument("overrides", nargs=-1, type=click.UNPROCESSED)
@click.option("--show-config", is_flag=True, help="Validate and print resolved configuration only.")
def main(command, overrides, show_config):
    routes = {
        "train": ("train_oof", "train_oof"),
        "train-yolo": ("train_oof", "train_yolo_roboflow"),
        "infer": ("video_to_csv", "video_to_csv"),
        "infer-cpu": ("video_to_csv", "laptop_cpu"),
        "track-eval": ("evaluate_tracks", "evaluate_tracks"),
        "ocr": ("text_prototype", "text_prototype"),
        "export": ("export_model", "export"),
        "export-yolo": ("export_yolo", "export_yolo"),
        "download": ("download", "download"),
        "serve-package": ("serving", "serving"),
        "serve-yolo-package": ("serving_yolo", "serving_yolo"),
    }
    module_name, config_name = routes[command]
    with initialize_config_module(config_module="configs", version_base=None):
        config = compose(config_name=config_name, overrides=list(overrides))
    if show_config:
        from omegaconf import OmegaConf

        click.echo(OmegaConf.to_yaml(config, resolve=True))
        return
    module = importlib.import_module(f"lenta_price_tag_scanner.{module_name}")
    module.main.__wrapped__(config)


if __name__ == "__main__":
    main()
