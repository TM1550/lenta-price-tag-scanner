"""Local OCR profiles must pass explicit directories and fail before model downloads."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from omegaconf import OmegaConf

from lenta_price_tag_scanner.text_prototype import TextReader


class LocalOcrAssetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_explicit_directories_are_forwarded(self):
        with tempfile.TemporaryDirectory() as folder:
            factory = Mock()
            cfg = OmegaConf.create(
                dict(
                    cpu_threads=1,
                    backend="paddle",
                    paddle_cache=folder,
                    paddle_rec="eslav_PP-OCRv5_mobile_rec",
                    text_detection_model_dir=folder,
                    text_recognition_model_dir=folder,
                )
            )
            with patch.dict("sys.modules", {"paddleocr": SimpleNamespace(PaddleOCR=factory)}):
                TextReader(cfg)
            self.assertEqual(
                factory.call_args.kwargs["text_detection_model_dir"], str(Path(folder).resolve())
            )
            self.assertEqual(
                factory.call_args.kwargs["text_recognition_model_dir"], str(Path(folder).resolve())
            )

    def test_missing_directory_does_not_create_reader(self):
        with tempfile.TemporaryDirectory() as folder:
            factory = Mock()
            cfg = OmegaConf.create(
                dict(
                    cpu_threads=1,
                    backend="paddle",
                    paddle_cache=folder,
                    text_detection_model_dir=str(Path(folder) / "missing"),
                )
            )
            with patch.dict("sys.modules", {"paddleocr": SimpleNamespace(PaddleOCR=factory)}):
                with self.assertRaises(FileNotFoundError):
                    TextReader(cfg)
            factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
