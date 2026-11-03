"""Manual labels must be explicit and retain original video groups."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

from omegaconf import OmegaConf
from PIL import Image

from lenta_price_tag_scanner.rebuild_efficient import main


class RebuildTests(unittest.TestCase):
    def test_manual_labels_and_grouping(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            source.mkdir()
            output = Path(directory) / "dataset"
            original, annotations = [], []
            for fold in range(5):
                for label in range(2):
                    filename = f"{fold}_{label}.png"
                    Image.new("RGB", (20, 20), (fold * 20, label * 40, 0)).save(source / filename)
                    original.append(
                        dict(crop=filename, video=f"/videos/video{fold}.mp4", fold=fold)
                    )
                    annotations.append(dict(image=filename, label=str(label), reviewed="true"))
            (source / "source_manifest.jsonl").write_text(
                "\n".join(json.dumps(row) for row in original)
            )
            with (source / "labels.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["image", "label", "reviewed"])
                writer.writeheader()
                writer.writerows(annotations)
            main.__wrapped__(OmegaConf.create(dict(source=str(source), output=str(output))))
            rows = json.loads((output / "efficient/manifest.json").read_text())
            folds = json.loads((output / "efficient/folds.json").read_text())
            self.assertEqual(len(rows), 10)
            self.assertTrue(all(row["source"] == "human_review" for row in rows))
            for fold in folds:
                self.assertEqual(len(fold["val"]), 2)
                self.assertFalse(set(fold["train"]) & set(fold["val"]))

    def test_unreviewed_not_silently_used(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            (source / "source_manifest.jsonl").write_text("")
            (source / "labels.csv").write_text("image,label,reviewed\nanything.png,1,false\n")
            output = source / "output"
            with self.assertRaises(ValueError):
                main.__wrapped__(OmegaConf.create(dict(source=str(source), output=str(output))))
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
