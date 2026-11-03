"""Inference integration contract: provenance, references and path containment."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from omegaconf import OmegaConf
from PIL import Image

from lenta_price_tag_scanner import text_prototype


class PrototypeTests(unittest.TestCase):
    def config(self, root):
        cfg = OmegaConf.load(Path(__file__).parents[1] / "configs/text_prototype.yaml")
        cfg.crop_manifest = str(root / "inputs.jsonl")
        cfg.output = str(root / "output")
        cfg.labels = str(root / "labels")
        return cfg

    def test_reference_is_only_used_for_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (10, 20)).save(root / "crop.png")
            record = {"image": "crop.png", "filename": "video.mp4", "frame_timestamp": 2.5}
            (root / "inputs.jsonl").write_text(json.dumps(record), encoding="utf-8")
            (root / "labels").mkdir()
            (root / "labels/crop.txt").write_text("reference secret", encoding="utf-8")
            cfg = self.config(root)
            with patch.object(text_prototype, "TextReader") as reader:
                reader.return_value.read.return_value = ("prediction", False)
                text_prototype.main.__wrapped__(cfg)
                image = reader.return_value.read.call_args.args[0]
                self.assertIsInstance(image, Image.Image)
                self.assertEqual(image.size, (20, 10))
                self.assertEqual(len(reader.return_value.read.call_args.args), 1)
            row = json.loads((root / "output/predictions.jsonl").read_text())
            self.assertEqual(row["text"], "prediction")
            self.assertEqual(row["source_observation"], record)
            self.assertEqual(row["video"], "video.mp4")
            summary = json.loads((root / "output/summary.json").read_text())
            self.assertEqual(summary["evaluated"], 1)
            self.assertGreater(summary["cer"], 0)

    def test_manifest_cannot_escape_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "inputs.jsonl").write_text(
                json.dumps({"image": "../outside.png"}), encoding="utf-8"
            )
            with (
                self.assertRaises(ValueError),
                patch.object(text_prototype, "TextReader") as reader,
            ):
                text_prototype.main.__wrapped__(self.config(root))
            reader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
