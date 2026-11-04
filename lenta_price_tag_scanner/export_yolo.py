"""Export a copied YOLO checkpoint and verify ONNX predictions on a real image."""

import shutil

import hydra

from .build_datasets import checksum, resolve, write_json
from .data_loader import download_data_with_dvc


@hydra.main(version_base=None, config_path="../configs", config_name="export_yolo")
def main(cfg):
    import numpy as np
    import onnx
    from ultralytics import YOLO

    download_data_with_dvc(cfg.dvc_targets, cfg.pull_dvc)
    output = resolve(cfg.output)
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = resolve(cfg.checkpoint)
    copied = output / "model.pt"
    shutil.copy2(checkpoint, copied)
    model = YOLO(str(copied))
    exported = model.export(
        format="onnx",
        imgsz=cfg.imgsz,
        device="cpu",
        dynamic=True,
        simplify=False,
        opset=cfg.opset,
        half=False,
    )
    onnx.checker.check_model(str(exported))
    options = dict(imgsz=cfg.imgsz, device="cpu", conf=cfg.conf, iou=cfg.iou, verbose=False)
    expected = model.predict(str(resolve(cfg.example)), **options)[0].boxes.data.cpu().numpy()
    actual = (
        YOLO(str(exported), task="detect")
        .predict(str(resolve(cfg.example)), **options)[0]
        .boxes.data.cpu()
        .numpy()
    )
    if not len(expected):
        raise ValueError("Parity example must contain detections")
    np.testing.assert_allclose(actual, expected, rtol=cfg.rtol, atol=cfg.atol)
    write_json(
        output / "export_report.json",
        dict(
            checkpoint_sha256=checksum(checkpoint),
            onnx_sha256=checksum(resolve(exported)),
            detections=len(actual),
            max_absolute_error=float(np.abs(actual - expected).max()),
            input="RGB float32 NCHW /255; letterbox to imgsz",
            output="Raw xywh pixels and class confidences; requires NMS and inverse letterbox",
            imgsz=cfg.imgsz,
            opset=cfg.opset,
        ),
    )


if __name__ == "__main__":
    main()
