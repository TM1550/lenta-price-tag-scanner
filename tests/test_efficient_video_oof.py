"""GPU admission must never select GPU1 or compete with an existing process."""

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner
from omegaconf import OmegaConf


class EfficientVideoAdmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location(
            "evaluate_efficient_oof",
            Path(__file__).resolve().parents[1] / "scripts/server/evaluate_efficient_oof.py",
        )
        cls.runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.runner)

    def test_incorrect_visibility_is_rejected_before_any_gpu_query(self):
        for visible in ("", "1", "0,1"):
            with self.subTest(visible=visible):
                with patch.dict(os.environ, CUDA_VISIBLE_DEVICES=visible):
                    with patch.object(self.runner.subprocess, "check_output") as query:
                        with self.assertRaises(RuntimeError):
                            self.runner.gpu_sample()
                        query.assert_not_called()

    def test_both_queries_are_restricted_to_physical_gpu_zero(self):
        with patch.dict(os.environ, CUDA_VISIBLE_DEVICES="0"):
            with patch.object(self.runner.subprocess, "check_output", side_effect=["0, 32202", ""]):
                sample = self.runner.gpu_sample()
                self.assertEqual(sample["free_mib"], 32202)
            with patch.object(
                self.runner.subprocess, "check_output", side_effect=["0, 32202", ""]
            ) as query:
                self.runner.gpu_sample()
                for call in query.call_args_list:
                    command = call.args[0]
                    self.assertEqual(command[command.index("-i") + 1], "0")

    def test_memory_holder_compute_load_and_insufficient_memory_prevent_admission(self):
        samples = [
            dict(utilization=0, free_mib=30000, processes="123"),
            dict(utilization=1, free_mib=30000, processes=""),
            dict(utilization=0, free_mib=15000, processes=""),
        ]
        resources = self.runner.runner_config(idle_seconds=0).resources
        for sample in samples:
            with self.subTest(sample=sample), tempfile.TemporaryDirectory() as directory:
                with patch.object(self.runner, "gpu_sample", return_value=sample):
                    with self.assertRaises(RuntimeError):
                        self.runner.admit_gpu(Path(directory), resources)
                self.assertTrue((Path(directory) / "gpu_admission.json").is_file())

    def test_invalid_resource_overrides_fail_before_gpu_or_filesystem_work(self):
        for override in ("resources.poll_seconds=0", "resources.min_gpu_free_mib=0"):
            with self.subTest(override=override), patch.object(self.runner, "gpu_sample") as query:
                result = CliRunner().invoke(
                    self.runner.main,
                    [
                        "--baseline",
                        "baseline",
                        "--efficient-assets",
                        "assets",
                        "--output",
                        "unused",
                        "--source-commit",
                        "test",
                        override,
                    ],
                )
                self.assertEqual(result.exit_code, 2, result.output)
                query.assert_not_called()

    def test_prepare_only_preserves_folds_and_overrides_without_gpu_queries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            baseline = root / "baseline"
            assets = root / "efficient"
            output = root / "prepared"
            baseline.mkdir()
            assets.mkdir()
            models = {}
            previous = []
            videos = [f"video_{fold}.mp4" for fold in range(5)]
            for fold, video in enumerate(videos):
                model = assets / f"fold_{fold}.pt"
                model.write_bytes(f"efficient-{fold}".encode())
                models[str(fold)] = dict(
                    file=model.name,
                    held_out_video=video,
                    train_videos=[other for other in videos if other != video],
                    sha256=self.runner.checksum(model),
                )
                detector = root / f"yolo/fold_{fold}/training/weights/last.pt"
                detector.parent.mkdir(parents=True)
                detector.write_bytes(f"yolo-{fold}".encode())
                training = detector.parents[2] / "train.txt"
                training.write_text("train-frame.jpg\n")
                previous.append(
                    dict(
                        fold=fold,
                        video=video,
                        checkpoint=str(detector),
                        checkpoint_sha256=self.runner.checksum(detector),
                        train_list_sha256=self.runner.checksum(training),
                    )
                )
                config_file = baseline / f"fold_{fold}/pipeline/config.yaml"
                config_file.parent.mkdir(parents=True)
                config_file.write_text("{}\n")
            (assets / "models.json").write_text(json.dumps(dict(models=models)))
            (baseline / "oof_provenance.json").write_text(json.dumps(previous))
            (baseline / "report.json").write_text("{}")
            (root / "SOURCE_MANIFEST.json").write_text("{}")
            (root / "project").mkdir()
            with patch.dict(os.environ, LENTA_WORKSPACE=str(root)):
                with patch.object(self.runner, "gpu_sample") as query:
                    with patch.object(self.runner.shutil, "disk_usage") as disk:
                        disk.return_value.free = 100 * self.runner.BYTES_PER_GIB
                        result = CliRunner().invoke(
                            self.runner.main,
                            [
                                "--baseline",
                                str(baseline),
                                "--efficient-assets",
                                str(assets),
                                "--output",
                                str(output),
                                "--source-commit",
                                "test-commit",
                                "--prepare-only",
                                "--idle-seconds",
                                "0",
                                "pipeline.efficient_rank_weight=0.75",
                                "resources.min_gpu_free_mib=17000",
                            ],
                        )
                    self.assertEqual(result.exit_code, 0, result.output or str(result.exception))
                    query.assert_not_called()
            experiment = json.loads((output / "experiment.json").read_text())
            self.assertEqual(experiment["rank_weight"], 0.75)
            self.assertEqual(experiment["source_commit"], "test-commit")
            self.assertEqual([job["fold"] for job in experiment["jobs"]], list(range(5)))
            settings = OmegaConf.load(output / "runner_config.yaml")
            self.assertEqual(settings.resources.idle_seconds, 0)
            self.assertTrue(
                self.runner.gpu_occupied(
                    dict(utilization=0, free_mib=16500, processes=""),
                    settings.resources,
                )
            )
            for fold in range(5):
                config = OmegaConf.load(output / f"pipeline_fold_{fold}.yaml")
                self.assertEqual(config.efficient_rank_weight, 0.75)
                self.assertEqual(Path(config.efficient_weights), assets / f"fold_{fold}.pt")
                self.assertIsNone(config.resume_tracks)


if __name__ == "__main__":
    unittest.main()
