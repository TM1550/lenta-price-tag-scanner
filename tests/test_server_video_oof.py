"""Server preparation must remain GPU-free and preserve the held-out video recipe."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from omegaconf import OmegaConf

from lenta_price_tag_scanner.server_video_oof import confined, gpu_preflight, plan


class ServerPlanTests(unittest.TestCase):
    def test_plan_does_not_launch_programs_and_uses_distinct_fold_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            dataset, videos = root / "dataset", root / "videos"
            (dataset / "yolo").mkdir(parents=True)
            videos.mkdir()
            rows = [dict(fold=i, video=f"video_{i}.mp4") for i in range(5)]
            (dataset / "yolo/manifest.json").write_text(json.dumps(rows))
            for row in rows:
                (videos / row["video"]).touch()
                (videos / Path(row["video"]).with_suffix(".csv")).touch()
            cfg = OmegaConf.create(
                dict(
                    dataset=str(dataset),
                    videos=str(videos),
                    output=str(root / "results/run"),
                    plan_directory=str(root / "plan"),
                    expected_folds=5,
                    qwen_model=str(root / "qwen"),
                    mlflow_uri=None,
                )
            )
            with patch.dict(os.environ, LENTA_WORKSPACE=str(root)):
                with patch("lenta_price_tag_scanner.server_video_oof.validate_dataset"):
                    with patch("subprocess.run") as launch:
                        previous = Path.cwd()
                        try:
                            os.chdir(root)
                            result = plan(cfg)
                        finally:
                            os.chdir(previous)
                        launch.assert_not_called()
            self.assertEqual(len(result["jobs"]), 5)
            self.assertEqual(len({job["checkpoint"] for job in result["jobs"]}), 5)
            for job in result["jobs"]:
                self.assertNotIn(job["video"], job["train_videos"])
                pipeline = OmegaConf.load(job["pipeline_config"])
                self.assertEqual(pipeline.pattern, job["video"])
                self.assertEqual(pipeline.yolo_weights, job["checkpoint"])
                self.assertTrue(pipeline.vlm_fallback)
                self.assertTrue(pipeline.ocr.local_files_only)
                self.assertIsNone(pipeline.efficient_weights)

    def test_preparation_environment_cannot_start_gpu_job(self):
        with patch.dict(os.environ, CUDA_VISIBLE_DEVICES=""):
            with patch("subprocess.run") as launch:
                with self.assertRaisesRegex(RuntimeError, "Explicit GPU0"):
                    gpu_preflight()
                launch.assert_not_called()

    def test_paths_outside_isolated_workspace_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            with patch.dict(os.environ, LENTA_WORKSPACE=str(root)):
                with self.assertRaises(ValueError):
                    confined(root.parent / "unrelated")


if __name__ == "__main__":
    unittest.main()
