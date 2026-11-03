"""Video to OCR/VLM input crops, without running a text model or external API."""

import json
from pathlib import Path

import hydra

from .build_datasets import checksum, resolve, write_json
from .prepare_crops import run


@hydra.main(version_base=None, config_path="../configs", config_name="ocr_crops")
def main(cfg):
    run(cfg)
    output = resolve(cfg.output)
    selected = []
    for line in (output / "manifest.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if cfg.efficient_weights and row["suggestion"] != "positive":
            continue
        selected.append(
            dict(
                image=row["crop"],
                filename=Path(row["video"]).name,
                frame_timestamp=row["timestamp_ms"],
                bbox_xyxy=row["bbox"],
                frame_index=row["frame"],
                detector_score=row["yolo_conf"],
                classifier_score=row["teacher_score"],
                crop_sha256=checksum(output / row["crop"]),
                text=None,
                product_id=None,
            )
        )
    with (output / "ocr_inputs.jsonl").open("w", encoding="utf-8") as stream:
        for row in selected:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_json(
        output / "ocr_summary.json",
        dict(
            selected=len(selected),
            detector_sha256=checksum(resolve(cfg.yolo_weights)),
            classifier_sha256=(
                checksum(resolve(cfg.efficient_weights)) if cfg.efficient_weights else None
            ),
            note="Paths relative to this directory. Crops are not unique tracked price tags.",
        ),
    )
    print(f"Prepared {len(selected)} OCR/VLM inputs in {output}")


if __name__ == "__main__":
    main()
