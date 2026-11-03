"""Export trained EfficientNet to ONNX, check parity, optionally build TensorRT."""

import shutil
import subprocess

import hydra

from .build_datasets import checksum, resolve, write_json
from .classifier import load_classifier
from .data_loader import download_data_with_dvc


@hydra.main(version_base=None, config_path="../configs", config_name="export")
def main(cfg):
    import numpy as np
    import onnx
    import onnxruntime as runtime
    import torch
    from PIL import Image

    download_data_with_dvc(cfg.dvc_targets, cfg.pull_dvc)
    if cfg.format not in ("onnx", "tensorrt"):
        raise ValueError("format must be onnx or tensorrt")
    output = resolve(cfg.output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    model, transform, size = load_classifier(resolve(cfg.checkpoint))
    with Image.open(resolve(cfg.example)) as image:
        pixels = transform(image.convert("RGB")).unsqueeze(0)
    path = output / "model.onnx"
    torch.onnx.export(
        model,
        pixels,
        str(path),
        opset_version=cfg.opset,
        dynamo=False,
        input_names=["images"],
        output_names=["logits"],
        dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}},
    )
    onnx.checker.check_model(str(path))
    options = runtime.SessionOptions()
    options.intra_op_num_threads = 4
    session = runtime.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    parity = []
    for batch_size in (1, 2):
        batch = pixels.repeat(batch_size, 1, 1, 1)
        with torch.inference_mode():
            expected = model(batch).numpy()
        actual = session.run(["logits"], {"images": batch.numpy()})[0]
        np.testing.assert_allclose(actual, expected, rtol=cfg.rtol, atol=cfg.atol)
        parity.append(
            dict(batch=batch_size, max_absolute_error=float(np.abs(actual - expected).max()))
        )
    write_json(
        output / "export_report.json",
        dict(
            checkpoint_sha256=checksum(resolve(cfg.checkpoint)),
            onnx_sha256=checksum(path),
            size=size,
            input="RGB, ImageNet normalization, NCHW float32",
            output="logits; apply sigmoid",
            parity=parity,
            opset=cfg.opset,
        ),
    )
    if cfg.format == "tensorrt":
        executable = shutil.which("trtexec")
        if executable is None:
            raise RuntimeError(
                "ONNX verified, but trtexec is not installed; TensorRT was not built"
            )
        subprocess.run(
            [
                executable,
                f"--onnx={path}",
                f"--saveEngine={output / 'model.engine'}",
                f"--minShapes=images:1x3x{size}x{size}",
                f"--optShapes=images:1x3x{size}x{size}",
                f"--maxShapes=images:8x3x{size}x{size}",
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
