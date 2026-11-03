"""Fetch versioned artifacts or bootstrap the public original videos."""

import hydra

from .data_loader import download_data, download_data_with_dvc


@hydra.main(version_base=None, config_path="../configs", config_name="download")
def main(cfg):
    if cfg.source == "dvc":
        download_data_with_dvc(cfg.targets)
    elif cfg.source == "public":
        download_data(cfg.url, cfg.output, cfg.folder)
    else:
        raise ValueError("source must be dvc or public")


if __name__ == "__main__":
    main()
