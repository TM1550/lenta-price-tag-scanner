"""GPU admission must fail closed on busy devices or insufficient disk."""

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from lenta_price_tag_scanner.resource_queue import admission


class ResourceAdmissionTests(unittest.TestCase):
    def test_busy_gpu_rejects_admission(self):
        with patch("shutil.disk_usage", return_value=SimpleNamespace(free=20 * 1024**3)):
            with patch(
                "lenta_price_tag_scanner.resource_queue.gpu_preflight",
                side_effect=RuntimeError("GPU0 occupied"),
            ):
                self.assertFalse(admission(Path("."))[0])

    def test_low_disk_rejects_before_gpu_probe(self):
        with patch("shutil.disk_usage", return_value=SimpleNamespace(free=1024**3)):
            with patch("lenta_price_tag_scanner.resource_queue.gpu_preflight") as gpu:
                self.assertFalse(admission(Path("."))[0])
                gpu.assert_not_called()

    def test_free_gpu_and_disk_allow_admission(self):
        with patch("shutil.disk_usage", return_value=SimpleNamespace(free=20 * 1024**3)):
            with patch("lenta_price_tag_scanner.resource_queue.gpu_preflight"):
                self.assertTrue(admission(Path("."))[0])


if __name__ == "__main__":
    unittest.main()
