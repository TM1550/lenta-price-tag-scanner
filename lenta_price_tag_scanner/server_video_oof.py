"""Plan or explicitly execute fresh five-fold training and complete video OOF inference."""

import csv
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import hydra
from hydra import compose, initialize_config_module
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from .build_datasets import checksum, resolve, write_json
from .yolo_folds import validate_dataset


def confined(path):
    root = Path(os.environ["LENTA_WORKSPACE"]).resolve()
    path = Path(path).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Path outside isolated workspace: {path}")
    return path


def plan(cfg):
    dataset, videos = confined(resolve(cfg.dataset)), confined(resolve(cfg.videos))
    validate_dataset(dataset, cfg.expected_folds)
    rows = json.loads((dataset / "yolo/manifest.json").read_text(encoding="utf-8"))
    output = confined(resolve(cfg.output))
    planning = confined(resolve(cfg.plan_directory))
    planning.mkdir(parents=True, exist_ok=True)
    mlflow = (
        cfg.mlflow_uri
        or f"sqlite:///{confined(Path(os.environ['LENTA_WORKSPACE']) / 'artifacts/mlflow.db')}"
    )
    command = [
        sys.executable,
        "-m",
        "lenta_price_tag_scanner.commands",
        "train-yolo",
        "pull_dvc=false",
        f"dataset={dataset}",
        f"output={output / 'training'}",
        f"mlflow_uri={mlflow}",
    ]
    jobs = []
    if GlobalHydra.instance().is_initialized():
        pipeline = compose(config_name="video_to_csv")
    else:
        with initialize_config_module(config_module="configs", version_base=None):
            pipeline = compose(config_name="video_to_csv")
    pipeline = OmegaConf.create(OmegaConf.to_container(pipeline, resolve=True))
    for fold in range(cfg.expected_folds):
        held_out = {row["video"] for row in rows if row["fold"] == fold}
        if len(held_out) != 1:
            raise ValueError("Expected one video per fold")
        video = next(iter(held_out))
        if (
            not (videos / video).is_file()
            or not (videos / Path(video).with_suffix(".csv")).is_file()
        ):
            raise FileNotFoundError(f"Missing video/CSV: {video}")
        checkpoint = output / f"training/fold_{fold}/training/weights/last.pt"
        pipeline_output = output / f"fold_{fold}/pipeline"
        evaluation = output / f"fold_{fold}/evaluation"
        merged = OmegaConf.merge(
            pipeline,
            dict(
                videos=str(videos),
                pattern=video,
                yolo_weights=str(checkpoint),
                pull_dvc=False,
                output=str(pipeline_output),
                detector_device="cuda",
                top_k=2,
                efficient_weights=None,
                efficient_mode="off",
                vlm_fallback=True,
                vlm_name_override=True,
                vlm_limit=1000,
                catalogue_csv=str(confined(resolve("data/inference_assets/db_hack.csv"))),
                catalogue_db=str(output / f"fold_{fold}/catalogue.sqlite"),
                ocr=dict(vlm_model=str(confined(resolve(cfg.qwen_model))), local_files_only=True),
            ),
        )
        config_file = planning / f"pipeline_fold_{fold}.yaml"
        OmegaConf.save(merged, config_file)
        jobs.append(
            dict(
                fold=fold,
                video=video,
                train_videos=sorted({row["video"] for row in rows if row["fold"] != fold}),
                checkpoint=str(checkpoint),
                pipeline_output=str(pipeline_output),
                evaluation=str(evaluation),
                pipeline_config=str(config_file),
            )
        )
    result = dict(
        training_command=command,
        jobs=jobs,
        output=str(output),
        dataset_manifest_sha256=checksum(dataset / "yolo/manifest.json"),
        detector_initialization="fresh architecture YAML, no teacher weights",
        ocr="fixed Paddle CPU + pinned Qwen3-VL-4B GPU, exact-code catalogue",
        efficientnet="disabled as in final pipeline",
        execution_authorized=False,
    )
    inventory = Path(os.environ["LENTA_WORKSPACE"]) / "TRANSFER_CODE.json"
    if inventory.is_file():
        for item in json.loads(inventory.read_text()):
            if checksum(confined(inventory.parent / item["path"])) != item["sha256"]:
                raise ValueError(f"Code snapshot changed: {item['path']}")
        result["source_inventory_sha256"] = checksum(inventory)
    write_json(planning / "plan.json", result)
    return result


def gpu_preflight():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("Explicit GPU0 launch required; preparation environment hides all GPUs")
    result = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            "0",
            "--query-gpu=uuid,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    uuid, used, utilization = [item.strip() for item in result.stdout.strip().split(",")]
    processes = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if (
        float(used) > 512
        or float(utilization) > 5
        or any(line.startswith(uuid) for line in processes.splitlines())
    ):
        raise RuntimeError("GPU0 occupied; refusing launch. GPU1 is never used")


def provenance(job, dataset, expected_epochs):
    rows = json.loads((dataset / "yolo/manifest.json").read_text(encoding="utf-8"))
    training_dir = Path(job["checkpoint"]).parents[2]
    actual = {
        Path(line).resolve() for line in (training_dir / "train.txt").read_text().splitlines()
    }
    expected = {(dataset / row["image"]).resolve() for row in rows if row["fold"] != job["fold"]}
    if actual != expected or job["video"] in job["train_videos"]:
        raise ValueError("Training list differs from expected outer-fold split")
    with (training_dir / "training/results.csv").open(newline="") as stream:
        epochs = [
            {key.strip(): value for key, value in row.items()} for row in csv.DictReader(stream)
        ]
    if not epochs or int(float(epochs[-1]["epoch"])) != expected_epochs:
        raise ValueError("Fixed-epoch training did not finish")
    checkpoint = Path(job["checkpoint"])
    metrics = json.loads((Path(job["pipeline_output"]) / "metrics.json").read_text())
    digest = checksum(checkpoint)
    if metrics["yolo_sha256"] != digest:
        raise ValueError("Video inference used a different detector")
    return dict(
        job,
        checkpoint_sha256=digest,
        train_list_sha256=checksum(training_dir / "train.txt"),
        dataset_manifest_sha256=checksum(dataset / "yolo/manifest.json"),
        epochs_completed=expected_epochs,
        last_checkpoint_used=True,
    )


def execute(cfg, prepared):
    output = Path(prepared["output"])
    if output.exists():
        raise FileExistsError("Use a new output directory; existing experiments are preserved")
    if shutil.disk_usage(confined(output.parent)).free < cfg.min_free_gib * 1024**3:
        raise RuntimeError("Insufficient disk for complete OOF outputs")
    qwen = confined(resolve(cfg.qwen_model))
    index = qwen / "model.safetensors.index.json"
    if not index.is_file():
        raise FileNotFoundError("Pinned local Qwen snapshot has not been downloaded")
    shards = set(json.loads(index.read_text())["weight_map"].values())
    if any(not (qwen / name).is_file() for name in shards):
        raise FileNotFoundError("Incomplete Qwen snapshot")
    gpu_preflight()
    output.mkdir(parents=True)

    def run(arguments, name):
        with (output / f"{name}.log").open("w", encoding="utf-8") as log:
            subprocess.run(arguments, stdout=log, stderr=subprocess.STDOUT, check=True)

    run(prepared["training_command"], "training")
    receipts, reports, predictions = [], [], []
    from .evaluate_fields import read_rows

    for job in prepared["jobs"]:
        # Never compete with another job that started between pipeline stages.
        gpu_preflight()
        config_file = Path(job["pipeline_config"])
        run(
            [
                sys.executable,
                "-m",
                "lenta_price_tag_scanner.video_to_csv",
                "--config-path",
                str(config_file.parent),
                "--config-name",
                config_file.stem,
            ],
            f"pipeline_fold_{job['fold']}",
        )
        receipt = provenance(job, resolve(cfg.dataset), 100)
        run(
            [
                sys.executable,
                "-m",
                "lenta_price_tag_scanner.commands",
                "track-eval",
                f"videos={resolve(cfg.videos)}",
                f"pattern={job['video']}",
                f"tracking_run={job['pipeline_output']}",
                f"prediction_run={job['pipeline_output']}",
                f"output={job['evaluation']}",
            ],
            f"evaluation_fold_{job['fold']}",
        )
        report_path = Path(job["evaluation"]) / "report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report.update(
            oof=True,
            oof_provenance=receipt,
            note="Detector training-list/checkpoint provenance verified. Automatic track associations remain unreviewed; not an official hackathon score.",
        )
        write_json(report_path, report)
        receipts.append(receipt)
        reports.append(report)
        predictions.extend(read_rows(Path(job["pipeline_output"]) / "submission.csv"))
        write_json(output / "oof_provenance.json", receipts)
    counts = Counter()
    for report in reports:
        counts.update(correct=report["correct"], expected=report["expected"])
    summary = dict(
        counts,
        micro_correct_fill=counts["correct"] / counts["expected"],
        macro_correct_fill=sum(report["correct_fill"] for report in reports) / len(reports),
        per_video={job["video"]: report for job, report in zip(prepared["jobs"], reports)},
        oof=True,
        automatic_associations=True,
        manual_identity_review_completed=False,
        dataset_manifest_sha256=prepared["dataset_manifest_sha256"],
    )
    write_json(output / "report.json", summary)
    from .evaluate_tracks import write_csv

    write_csv(output / "submission_oof.csv", predictions)
    print(
        json.dumps({key: value for key, value in summary.items() if key != "per_video"}, indent=2)
    )


@hydra.main(version_base=None, config_path="../configs", config_name="server_video_oof")
def main(cfg):
    prepared = plan(cfg)
    if cfg.execute:
        execute(cfg, prepared)
    else:
        print(
            json.dumps(
                dict(
                    plan_directory=str(resolve(cfg.plan_directory)),
                    jobs=len(prepared["jobs"]),
                    executed=False,
                    message="Preparation only; no training/inference jobs launched",
                ),
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
