"""Re-evaluate existing held-out YOLO folds with video-matched EfficientNet weights."""

import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import click
from hydra import compose, initialize_config_module
from omegaconf import OmegaConf

from lenta_price_tag_scanner.build_datasets import checksum, write_json
from lenta_price_tag_scanner.evaluate_fields import read_rows
from lenta_price_tag_scanner.evaluate_tracks import write_csv

GPU_INDEX = "0"
BYTES_PER_GIB = 1024**3


def runner_config(overrides=(), idle_seconds=None):
    with initialize_config_module(config_module="configs", version_base=None):
        config = compose(config_name="evaluate_efficient_oof", overrides=list(overrides))
    if idle_seconds is not None:
        config.resources.idle_seconds = idle_seconds
    resources = config.resources
    if resources.idle_seconds < 0 or resources.poll_seconds <= 0:
        raise click.BadParameter("Idle seconds must be nonnegative; poll seconds must be positive")
    if resources.min_gpu_free_mib <= 0 or resources.min_disk_free_gib <= 0:
        raise click.BadParameter("Free GPU memory and disk requirements must be positive")
    if config.expected_folds < 2:
        raise click.BadParameter("At least two held-out folds are required")
    if not 0 <= config.pipeline.efficient_rank_weight <= 1:
        raise click.BadParameter("EfficientNet rank weight must be between zero and one")
    return config


def confined(path):
    root = Path(os.environ["LENTA_WORKSPACE"]).resolve()
    path = Path(path).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Path outside project workspace: {path}")
    return path


def gpu_sample():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != GPU_INDEX:
        raise RuntimeError("Only physical GPU0 may be visible")
    command = ["nvidia-smi", "-i", GPU_INDEX, "--format=csv,noheader,nounits"]
    values = (
        subprocess.check_output(command + ["--query-gpu=utilization.gpu,memory.free"], text=True)
        .strip()
        .split(",")
    )
    processes = subprocess.check_output(
        ["nvidia-smi", "-i", GPU_INDEX, "--query-compute-apps=pid", "--format=csv,noheader"],
        text=True,
    ).strip()
    return dict(utilization=int(values[0]), free_mib=int(values[1]), processes=processes)


def gpu_occupied(sample, resources):
    return bool(
        sample["utilization"]
        or sample["free_mib"] < resources.min_gpu_free_mib
        or sample["processes"]
    )


def admit_gpu(output, resources):
    started = time.monotonic()
    samples = []
    while True:
        sample = gpu_sample()
        samples.append(dict(utc=datetime.now(timezone.utc).isoformat(), **sample))
        write_json(output / "gpu_admission.json", samples)
        if gpu_occupied(sample, resources):
            raise RuntimeError("GPU0 occupied; no launch and no competition processes modified")
        elapsed = time.monotonic() - started
        if elapsed >= resources.idle_seconds:
            return
        time.sleep(min(resources.poll_seconds, resources.idle_seconds - elapsed))


@click.command()
@click.option("--baseline", required=True, type=click.Path(path_type=Path))
@click.option("--efficient-assets", required=True, type=click.Path(path_type=Path))
@click.option("--output", required=True, type=click.Path(path_type=Path))
@click.option("--source-commit", required=True)
@click.option("--prepare-only", is_flag=True, help="Prepare fold configs without querying the GPU.")
@click.option("--idle-seconds", type=click.IntRange(min=0), default=None)
@click.argument("overrides", nargs=-1)
def main(baseline, efficient_assets, output, source_commit, prepare_only, idle_seconds, overrides):
    """Evaluate held-out folds; trailing KEY=VALUE arguments override Hydra settings."""
    settings = runner_config(overrides, idle_seconds)
    baseline = confined(baseline)
    assets = confined(efficient_assets)
    output = confined(output)
    output.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output).free < settings.resources.min_disk_free_gib * BYTES_PER_GIB:
        raise RuntimeError(
            f"At least {settings.resources.min_disk_free_gib} GiB free disk required"
        )
    OmegaConf.save(settings, output / "runner_config.yaml")
    manifest = json.loads((assets / "models.json").read_text())
    previous = json.loads((baseline / "oof_provenance.json").read_text())
    if {item["fold"] for item in previous} != set(range(settings.expected_folds)) or set(
        manifest["models"]
    ) != {str(fold) for fold in range(settings.expected_folds)}:
        raise ValueError(f"{settings.expected_folds} verified folds are required")
    previous.sort(key=lambda item: item["fold"])
    defaults = settings.pipeline
    defaults = OmegaConf.create(OmegaConf.to_container(defaults, resolve=True))
    jobs = []
    for old in previous:
        fold = old["fold"]
        model = manifest["models"][str(fold)]
        if model["held_out_video"] != old["video"] or old["video"] in model["train_videos"]:
            raise ValueError("EfficientNet held-out video mismatch")
        efficient = confined(assets / model["file"])
        detector = confined(old["checkpoint"])
        if checksum(efficient) != model["sha256"]:
            raise ValueError("EfficientNet checksum mismatch")
        if checksum(detector) != old["checkpoint_sha256"]:
            raise ValueError("YOLO checksum mismatch")
        training_list = detector.parents[2] / "train.txt"
        if checksum(training_list) != old["train_list_sha256"]:
            raise ValueError("Original YOLO train list changed")
        old_config = confined(baseline / f"fold_{fold}/pipeline/config.yaml")
        config = OmegaConf.merge(defaults, OmegaConf.load(old_config))
        config.pull_dvc = False
        config.efficient_weights = str(efficient)
        config.efficient_mode = "rank"
        config.efficient_rank_weight = settings.pipeline.efficient_rank_weight
        config.yolo_weights = str(detector)
        config.resume_tracks = None
        config.output = str(output / f"fold_{fold}/pipeline")
        config.catalogue_db = str(output / f"fold_{fold}/catalogue.sqlite")
        # Baseline OCR paths were relative to the old project's working directory.
        old_project = confined(Path(os.environ["LENTA_WORKSPACE"]) / "project")
        for key in ("paddle_cache", "text_detection_model_dir", "text_recognition_model_dir"):
            if config.ocr.get(key):
                config.ocr[key] = str(confined(old_project / config.ocr[key]))
        config.ocr.local_files_only = True
        config_file = output / f"pipeline_fold_{fold}.yaml"
        OmegaConf.save(config, config_file)
        jobs.append(
            dict(
                fold=fold,
                video=old["video"],
                config=str(config_file),
                yolo=old,
                efficient=model,
                efficient_checkpoint=str(efficient),
            )
        )
    provenance = dict(
        source_commit=source_commit,
        efficient_mode="rank",
        rank_weight=settings.pipeline.efficient_rank_weight,
        source_inventory_sha256=checksum(assets.parent / "SOURCE_MANIFEST.json"),
        efficient_manifest=manifest,
        baseline_report_sha256=checksum(baseline / "report.json"),
        gpu=int(GPU_INDEX),
        manual_identity_review_completed=False,
        jobs=jobs,
    )
    write_json(output / "experiment.json", provenance)
    if prepare_only:
        print(json.dumps(dict(prepared_folds=len(jobs), executed=False)))
        return
    write_json(output / "status.json", dict(state="gpu_admission", completed_folds=[]))
    admit_gpu(output, settings.resources)
    reports = []
    predictions = []
    for job in jobs:
        fold = job["fold"]
        destination = output / f"fold_{fold}"
        destination.mkdir(exist_ok=True)
        report_path = destination / "evaluation/report.json"
        if not report_path.is_file():
            sample = gpu_sample()
            if gpu_occupied(sample, settings.resources):
                raise RuntimeError("GPU0 no longer free between folds; existing results preserved")
            pipeline = destination / "pipeline"
            if not (pipeline / "metrics.json").is_file():
                if pipeline.exists():
                    raise RuntimeError("Incomplete pipeline exists; inspect before retrying")
                write_json(
                    output / "status.json",
                    dict(
                        state="pipeline",
                        fold=fold,
                        video=job["video"],
                        completed_folds=[item["fold"] for item in reports],
                    ),
                )
                with (destination / "pipeline.log").open("w") as stream:
                    subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "lenta_price_tag_scanner.video_to_csv",
                            "--config-path",
                            str(output),
                            "--config-name",
                            f"pipeline_fold_{fold}",
                        ],
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
            metrics = json.loads((pipeline / "metrics.json").read_text())
            if (
                metrics["efficient_mode"] != "rank"
                or metrics["efficient_sha256"] != job["efficient"]["sha256"]
                or metrics["yolo_sha256"] != job["yolo"]["checkpoint_sha256"]
            ):
                raise ValueError("Inference model provenance mismatch")
            scored = 0
            with (pipeline / "filter_audit.jsonl").open() as stream:
                for line in stream:
                    scored += json.loads(line).get("classifier_score") is not None
            if not scored:
                raise ValueError("No crops were actually scored by EfficientNet")
            with (destination / "evaluation.log").open("w") as stream:
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "lenta_price_tag_scanner.commands",
                        "track-eval",
                        f"videos={OmegaConf.load(job['config']).videos}",
                        f"pattern={job['video']}",
                        f"tracking_run={pipeline}",
                        f"prediction_run={pipeline}",
                        f"output={destination / 'evaluation'}",
                    ],
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            report = json.loads(report_path.read_text())
            report.update(
                fold=fold,
                oof=True,
                efficient_mode="rank",
                model_provenance=job,
                classifier_scored_crops=scored,
                manual_identity_review_completed=False,
            )
            write_json(report_path, report)
        report = json.loads(report_path.read_text())
        reports.append(report)
        predictions.extend(read_rows(destination / "pipeline/submission.csv"))
        counts = Counter()
        for item in reports:
            counts.update(correct=item["correct"], expected=item["expected"])
        summary = dict(
            counts,
            micro_correct_fill=counts["correct"] / counts["expected"],
            macro_correct_fill=sum(item["correct_fill"] for item in reports) / len(reports),
            per_video={jobs[item["fold"]]["video"]: item for item in reports},
            oof=True,
            efficient_mode="rank",
            source_commit=source_commit,
            automatic_associations=True,
            manual_identity_review_completed=False,
        )
        write_json(output / "report_partial.json", summary)
        write_json(
            output / "status.json",
            dict(
                state="fold_completed",
                fold=fold,
                completed_folds=[item["fold"] for item in reports],
            ),
        )
    baseline_report = json.loads((baseline / "report.json").read_text())
    summary["baseline_micro_correct_fill"] = baseline_report["micro_correct_fill"]
    summary["delta_micro_percentage_points"] = 100 * (
        summary["micro_correct_fill"] - baseline_report["micro_correct_fill"]
    )
    write_json(output / "report.json", summary)
    write_csv(output / "submission_oof.csv", predictions)
    write_json(
        output / "status.json",
        dict(state="complete", completed_folds=list(range(settings.expected_folds))),
    )
    print(
        json.dumps({key: value for key, value in summary.items() if key != "per_video"}, indent=2)
    )


if __name__ == "__main__":
    main()
