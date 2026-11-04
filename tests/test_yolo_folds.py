"""OOF membership, portability and annotation integrity must survive dataset assembly."""

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from click.testing import CliRunner
from omegaconf import OmegaConf
from PIL import Image

from lenta_price_tag_scanner.build_datasets import checksum
from lenta_price_tag_scanner.commands import main as cli
from lenta_price_tag_scanner.train_oof import yolo
from lenta_price_tag_scanner.yolo_folds import main, make_folds, validate_dataset


class YoloFoldTests(unittest.TestCase):
    def test_every_video_is_held_out_once(self):
        rows = [
            dict(image=f"{fold}_{frame}.jpg", video=f"{fold}.mp4", fold=fold)
            for fold in range(5)
            for frame in range(fold + 1)
        ]
        folds = make_folds(rows, 5)
        validation = [path for fold in folds for path in fold["val"]]
        self.assertCountEqual(validation, [row["image"] for row in rows])
        for fold in folds:
            self.assertFalse(set(fold["train"]) & set(fold["val"]))

    def test_video_cannot_leak_across_folds(self):
        rows = [dict(image=f"{fold}.jpg", video=f"{fold}.mp4", fold=fold) for fold in range(5)]
        rows.append(dict(image="leak.jpg", video="0.mp4", fold=1))
        with self.assertRaisesRegex(ValueError, "multiple folds"):
            make_folds(rows, 5)

    def test_missing_video_cannot_silently_reduce_fold_count(self):
        rows = [dict(image=f"{fold}.jpg", video=f"{fold}.mp4", fold=fold) for fold in range(4)]
        with self.assertRaises(ValueError):
            make_folds(rows, 5)

    def test_build_relocation_and_label_tamper_detection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            source, output = root / "source", root / "assembled"
            (source / "train/images").mkdir(parents=True)
            (source / "train/labels").mkdir(parents=True)
            rows = []
            for index in range(6):
                image = source / f"train/images/{index}.png"
                label = source / f"train/labels/{index}.txt"
                Image.new("RGB", (40, 40), (index * 30, 0, 0)).save(image)
                label.write_text("0 0.5 0.5 0.5 0.5\n" if index < 5 else "", encoding="utf-8")
                rows.append(
                    dict(
                        image=image.relative_to(source).as_posix(),
                        label=label.relative_to(source).as_posix(),
                        fold=index % 5,
                        video=f"{index % 5}.mp4",
                        frame_index=index,
                        timestamp_ms=index * 500,
                        source_image_sha256=checksum(image),
                        cleaned_label_sha256=checksum(label),
                        kept_boxes=int(index < 5),
                    )
                )
            (source / "manifest.json").write_text(json.dumps(rows), encoding="utf-8")
            (source / "data.yaml").write_text("names: [price_tag]\n", encoding="utf-8")
            (source / "COMPLETE.txt").touch()
            (source / "report.json").write_text("{}", encoding="utf-8")
            for name in ("source_README.dataset.txt", "source_README.roboflow.txt"):
                (source / name).write_text("fixture", encoding="utf-8")
            main.__wrapped__(
                OmegaConf.create(
                    dict(
                        source=str(source), output=str(output), expected_folds=5, exclude_empty=True
                    )
                )
            )
            report = json.loads((output / "validation.json").read_text())
            self.assertEqual(
                (report["images"], report["boxes"], report["excluded_empty"]), (5, 5, 1)
            )
            relocated = root / "relocated"
            shutil.copytree(output, relocated)
            self.assertEqual(validate_dataset(relocated)["images"], 5)
            label = next((relocated / "yolo/labels").rglob("*.txt"))
            label.write_text("", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Changed label"):
                validate_dataset(relocated)

    def test_new_cli_selects_only_the_new_dataset(self):
        result = CliRunner().invoke(cli, ["train-yolo", "--show-config"])
        self.assertEqual(result.exit_code, 0, result.output)
        cfg = OmegaConf.create(result.output)
        self.assertEqual(list(cfg.folds), list(range(5)))
        self.assertEqual(list(cfg.dvc_targets), ["data/yolo_roboflow_5fold_v1.dvc"])
        self.assertTrue(cfg.verify_dataset)
        self.assertEqual(cfg.oof.kind, "yolo")

    def test_training_uses_fresh_models_and_last_checkpoint_per_fold(self):
        result = CliRunner().invoke(cli, ["train-yolo", "--show-config"])
        cfg = OmegaConf.create(result.output)
        training_model, trained = Mock(), Mock()
        trained.val.return_value = SimpleNamespace(results_dict={"metrics/mAP50(B)": 0.5})
        trained.predict.return_value = [SimpleNamespace(boxes=SimpleNamespace(data=Mock()))]
        trained.predict.return_value[0].boxes.data.cpu.return_value.tolist.return_value = []
        constructor = Mock(side_effect=[training_model, trained, training_model, trained])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [dict(image="yolo/images/video/frame.jpg", video="video.mp4", fold=0)]
            with patch.dict(
                "sys.modules", {"mlflow": Mock(), "ultralytics": SimpleNamespace(YOLO=constructor)}
            ):
                with patch(
                    "lenta_price_tag_scanner.train_oof.prepare", return_value=(root, rows, rows)
                ):
                    for fold in (0, 1):
                        output = root / f"fold_{fold}"
                        output.mkdir()
                        yolo(cfg, fold, output)
            initialized = [call.args[0] for call in constructor.call_args_list]
            self.assertEqual(initialized[::2], ["yolov8n.yaml", "yolov8n.yaml"])
            self.assertEqual(initialized[1], str(root / "fold_0/training/weights/last.pt"))
            self.assertEqual(initialized[3], str(root / "fold_1/training/weights/last.pt"))
            for call in training_model.train.call_args_list:
                self.assertFalse(call.kwargs["pretrained"])
                self.assertFalse(call.kwargs["val"])
                self.assertEqual(call.kwargs["patience"], 0)
                self.assertEqual(call.kwargs["optimizer"], "AdamW")
                self.assertEqual(call.kwargs["epochs"], 100)
            self.assertEqual(
                [call.kwargs["seed"] for call in training_model.train.call_args_list], [42, 43]
            )


if __name__ == "__main__":
    unittest.main()
