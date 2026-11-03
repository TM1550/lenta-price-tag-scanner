"""One CLI; Click dispatches Hydra overrides without importing training dependencies."""

import importlib

import click
from hydra import compose, initialize_config_module


@click.command(context_settings={"ignore_unknown_options": True})
@click.argument(
    "command", type=click.Choice(["train", "infer", "ocr", "export", "download", "serve-package"])
)
@click.argument("overrides", nargs=-1, type=click.UNPROCESSED)
@click.option("--show-config", is_flag=True, help="Validate and print resolved configuration only.")
def main(command, overrides, show_config):
    routes = {
        "train": ("train_oof", "train_oof"),
        "infer": ("video_to_csv", "video_to_csv"),
        "ocr": ("text_prototype", "text_prototype"),
        "export": ("export_model", "export"),
        "download": ("download", "download"),
        "serve-package": ("serving", "serving"),
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
