"""Deployment routing and classifier ranking must preserve evaluation provenance."""

import json
import tempfile
import unittest
from pathlib import Path

from lenta_price_tag_scanner.build_datasets import checksum
from lenta_price_tag_scanner.detector_assets import classifier_selection, detector_for_video


class DetectorAssetsTest(unittest.TestCase):
    def test_rank_retains_low_probability_crop(self):
        self.assertEqual(classifier_selection(10, 0, "rank", 0.5, 0.5), (5, False))
        self.assertEqual(classifier_selection(10, 0, "filter", 0.5, 0.5), (10, True))
        self.assertEqual(classifier_selection(10, 0, "off", 0.5, 0.5), (10, False))

    def test_fold_routing_and_checksum(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "fold.pt"
            checkpoint.write_bytes(b"checkpoint")
            manifest = root / "models.json"
            manifest.write_text(
                json.dumps(
                    {
                        "held.mp4": dict(
                            model="fold.pt",
                            sha256=checksum(checkpoint),
                            fold=2,
                            train_videos=["other.mp4"],
                        )
                    }
                ),
                encoding="utf-8",
            )
            config = dict(yolo_model_map=str(manifest), yolo_default_weights=str(checkpoint))
            self.assertEqual(detector_for_video(config, "held.mp4")["fold"], 2)
            self.assertEqual(detector_for_video(config, "new.mp4")["source"], "deployment_default")
            checkpoint.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "checksum"):
                detector_for_video(config, "held.mp4")
            config["yolo_weights"] = str(checkpoint)
            self.assertEqual(detector_for_video(config, "held.mp4")["source"], "explicit")

    def test_reject_invalid_classifier_configuration(self):
        with self.assertRaises(ValueError):
            classifier_selection(1, None, "unknown", 0.5, 0.5)
        with self.assertRaises(ValueError):
            classifier_selection(1, None, "rank", 0.5, 2)
