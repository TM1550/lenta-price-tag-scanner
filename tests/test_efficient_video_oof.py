"""GPU admission must never select GPU1 or compete with an existing process."""

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    "evaluate_efficient_oof",
    Path(__file__).resolve().parents[1] / "scripts/server/evaluate_efficient_oof.py",
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class EfficientVideoAdmissionTests(unittest.TestCase):
    def test_incorrect_visibility_is_rejected_before_any_gpu_query(self):
        for visible in ("", "1", "0,1"):
            with self.subTest(visible=visible):
                with patch.dict(os.environ, CUDA_VISIBLE_DEVICES=visible):
                    with patch.object(runner.subprocess, "check_output") as query:
                        with self.assertRaises(RuntimeError):
                            runner.gpu_sample()
                        query.assert_not_called()

    def test_both_queries_are_restricted_to_physical_gpu_zero(self):
        with patch.dict(os.environ, CUDA_VISIBLE_DEVICES="0"):
            with patch.object(runner.subprocess, "check_output", side_effect=["0, 32202", ""]):
                sample = runner.gpu_sample()
                self.assertEqual(sample["free_mib"], 32202)
            with patch.object(
                runner.subprocess, "check_output", side_effect=["0, 32202", ""]
            ) as query:
                runner.gpu_sample()
                for call in query.call_args_list:
                    command = call.args[0]
                    self.assertEqual(command[command.index("-i") + 1], "0")

    def test_memory_holder_compute_load_and_insufficient_memory_prevent_admission(self):
        samples = [
            dict(utilization=0, free_mib=30000, processes="123"),
            dict(utilization=1, free_mib=30000, processes=""),
            dict(utilization=0, free_mib=15000, processes=""),
        ]
        for sample in samples:
            with self.subTest(sample=sample), tempfile.TemporaryDirectory() as directory:
                with patch.object(runner, "gpu_sample", return_value=sample):
                    with self.assertRaises(RuntimeError):
                        runner.admit_gpu(Path(directory), 0)
                self.assertTrue((Path(directory) / "gpu_admission.json").is_file())


if __name__ == "__main__":
    unittest.main()
