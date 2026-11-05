"""Re-evaluate existing held-out YOLO folds with video-matched EfficientNet weights."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from hydra import compose, initialize_config_module
from omegaconf import OmegaConf

from lenta_price_tag_scanner.build_datasets import checksum, write_json
from lenta_price_tag_scanner.evaluate_fields import read_rows
from lenta_price_tag_scanner.evaluate_tracks import write_csv


def confined(path):
    root = Path(os.environ["LENTA_WORKSPACE"]).resolve()
    path = Path(path).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Path outside project workspace: {path}")
    return path


def gpu_sample():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("Only physical GPU0 may be visible")
    command = ["nvidia-smi", "-i", "0", "--format=csv,noheader,nounits"]
    values = (
        subprocess.check_output(command + ["--query-gpu=utilization.gpu,memory.free"], text=True)
        .strip()
        .split(",")
    )
    processes = subprocess.check_output(
        ["nvidia-smi", "-i", "0", "--query-compute-apps=pid", "--format=csv,noheader"],
        text=True,
    ).strip()
    return dict(utilization=int(values[0]), free_mib=int(values[1]), processes=processes)


def admit_gpu(output, seconds):
    started = time.monotonic()
    samples = []
    while True:
        sample = gpu_sample()
        samples.append(dict(utc=datetime.now(timezone.utc).isoformat(), **sample))
        write_json(output / "gpu_admission.json", samples)
        if sample["utilization"] or sample["free_mib"] < 16000 or sample["processes"]:
            raise RuntimeError("GPU0 occupied; no launch and no competition processes modified")
        if time.monotonic() - started >= seconds:
            return
        time.sleep(min(5, seconds - (time.monotonic() - started)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--efficient-assets", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--idle-seconds", type=int, default=120)
    args = parser.parse_args()
    if args.idle_seconds < 0:
        parser.error("idle-seconds must be nonnegative")
    baseline = confined(args.baseline)
    assets = confined(args.efficient_assets)
    output = confined(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output).free < 12 * 1024**3:
        raise RuntimeError("At least 12 GiB free disk required")
    manifest = json.loads((assets / "models.json").read_text())
    previous = json.loads((baseline / "oof_provenance.json").read_text())
    if {item["fold"] for item in previous} != set(range(5)) or set(manifest["models"]) != {
        str(fold) for fold in range(5)
    }:
        raise ValueError("Five verified folds are required")
    previous.sort(key=lambda item: item["fold"])
    with initialize_config_module(config_module="configs", version_base=None):
        defaults = compose(config_name="video_to_csv")
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
        config.efficient_rank_weight = 0.5
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
        source_commit=args.source_commit,
        efficient_mode="rank",
        rank_weight=0.5,
        source_inventory_sha256=checksum(assets.parent / "SOURCE_MANIFEST.json"),
        efficient_manifest=manifest,
        baseline_report_sha256=checksum(baseline / "report.json"),
        gpu=0,
        manual_identity_review_completed=False,
        jobs=jobs,
    )
    write_json(output / "experiment.json", provenance)
    if args.prepare_only:
        print(json.dumps(dict(prepared_folds=len(jobs), executed=False)))
        return
    write_json(output / "status.json", dict(state="gpu_admission", completed_folds=[]))
    admit_gpu(output, args.idle_seconds)
    reports = []
    predictions = []
    for job in jobs:
        fold = job["fold"]
        destination = output / f"fold_{fold}"
        destination.mkdir(exist_ok=True)
        report_path = destination / "evaluation/report.json"
        if not report_path.is_file():
            sample = gpu_sample()
            if sample["utilization"] or sample["free_mib"] < 16000 or sample["processes"]:
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
            source_commit=args.source_commit,
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
    write_json(output / "status.json", dict(state="complete", completed_folds=list(range(5))))
    print(
        json.dumps({key: value for key, value in summary.items() if key != "per_video"}, indent=2)
    )


if __name__ == "__main__":
    main()
