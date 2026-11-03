"""Checkout/wheel paths and CLI dispatch must work without executing training."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from lenta_price_tag_scanner.build_datasets import resolve
from lenta_price_tag_scanner.commands import main
from lenta_price_tag_scanner.data_loader import download_data_with_dvc


class PackagingTests(unittest.TestCase):
    def test_config_includes_inference_and_no_training_dispatch(self):
        result = CliRunner().invoke(main, ["infer", "--show-config"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("yolo_weights:", result.output)
        self.assertIn("pull_dvc: true", result.output)

    def test_paths_resolve_from_working_directory(self):
        original = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                absolute = Path(directory).resolve() / "result"
                self.assertEqual(resolve("result"), absolute)
                self.assertEqual(resolve(absolute), absolute)
            finally:
                os.chdir(original)

    def test_disabling_dvc_does_not_access_repository(self):
        with patch("lenta_price_tag_scanner.data_loader.repository_root") as root:
            download_data_with_dvc([], enabled=False)
            root.assert_not_called()

    def test_dvc_empty_targets_fails_before_any_network(self):
        with self.assertRaises(ValueError):
            download_data_with_dvc([], enabled=True)


if __name__ == "__main__":
    unittest.main()
