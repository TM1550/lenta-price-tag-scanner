"""Extract review candidates. Run with python -m lenta_price_tag_scanner.prepare_crops."""

import hashlib
import json
import math
from pathlib import Path


def bucket(probability, low, high):
    """Teacher predictions are suggestions, never ground truth."""
    if probability is None:
        return "review"
    if probability <= low:
        return "negative"
    return "positive" if probability >= high else "review"


def run(cfg):
    import cv2
    from omegaconf import OmegaConf
    from ultralytics import YOLO

    def resolve(value):
        path = Path(value)
        return path.expanduser().resolve()

    videos = sorted(resolve(cfg.videos).glob(cfg.pattern))
    weights = resolve(cfg.yolo_weights)
    output = resolve(cfg.output)
    if not videos or not weights.is_file():
        raise ValueError("Videos or local YOLO weights not found")
    if not 0 <= cfg.low < cfg.high <= 1:
        raise ValueError("Require 0 <= low < high <= 1")
    if cfg.stride < 1 or cfg.min_side < 1 or cfg.max_frames < 0:
        raise ValueError("Invalid stride, min_side or max_frames")
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    teacher = None
    if cfg.efficient_weights:
        import timm
        import torch
        from torchvision import transforms

        checkpoint = torch.load(
            resolve(cfg.efficient_weights), map_location="cpu", weights_only=True
        )
        modern = "state_dict" in checkpoint
        architecture = checkpoint["architecture"] if modern else "efficientnet_b0"
        size = checkpoint["size"] if modern else cfg.crop_size
        if modern and (checkpoint.get("color") != "RGB" or checkpoint.get("positive_class") != 1):
            raise ValueError("Unsupported classifier checkpoint preprocessing")
        legacy_bgr = False if modern else cfg.legacy_bgr
        invert_probability = False if modern else cfg.invert_probability
        teacher = timm.create_model(architecture, pretrained=False, num_classes=1)
        teacher.load_state_dict(checkpoint["state_dict"] if modern else checkpoint)
        teacher.to(cfg.device).eval()
        transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    detector = YOLO(str(weights))
    output.mkdir(parents=True)
    OmegaConf.save(cfg, output / "config.yaml")
    counts = {"positive": 0, "negative": 0, "review": 0, "small": 0, "duplicate": 0}
    with (output / "manifest.jsonl").open("w", encoding="utf-8") as manifest:
        for fold, video in enumerate(videos):
            capture = cv2.VideoCapture(str(video))
            try:
                fps = capture.get(cv2.CAP_PROP_FPS)
                if not capture.isOpened() or not math.isfinite(fps) or fps <= 0:
                    raise ValueError(f"Cannot decode video or invalid FPS: {video}")
                seen = set()
                frame_index = -1
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    frame_index += 1
                    if cfg.max_frames and frame_index >= cfg.max_frames:
                        break
                    if frame_index % cfg.stride:
                        continue
                    result = detector.predict(
                        frame,
                        conf=cfg.conf,
                        iou=cfg.iou,
                        imgsz=cfg.imgsz,
                        device=cfg.device,
                        verbose=False,
                    )[0]
                    for detection_index, box in enumerate(result.boxes):
                        left, top, right, bottom = box.xyxy[0].cpu().tolist()
                        height, width = frame.shape[:2]
                        left, top = max(0, math.floor(left)), max(0, math.floor(top))
                        right, bottom = min(width, math.ceil(right)), min(height, math.ceil(bottom))
                        if min(right - left, bottom - top) < cfg.min_side:
                            counts["small"] += 1
                            continue
                        crop = frame[top:bottom, left:right]
                        digest = hashlib.sha256(crop.tobytes()).hexdigest()
                        if digest in seen:
                            counts["duplicate"] += 1
                            continue
                        seen.add(digest)
                        probability = None
                        if teacher is not None:
                            pixels = crop if legacy_bgr else cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                            with torch.inference_mode():
                                probability = float(
                                    torch.sigmoid(
                                        teacher(transform(pixels).unsqueeze(0).to(cfg.device))
                                    ).item()
                                )
                            if invert_probability:
                                probability = 1 - probability
                        group = bucket(probability, cfg.low, cfg.high)
                        relative = (
                            Path(group)
                            / video.stem
                            / f"{frame_index:08d}_{detection_index:03d}.jpg"
                        )
                        target = output / relative
                        target.parent.mkdir(parents=True, exist_ok=True)
                        success, encoded = cv2.imencode(".jpg", crop)
                        if not success:
                            raise RuntimeError(f"JPEG encoding failed: {target}")
                        encoded.tofile(str(target))
                        row = dict(
                            crop=relative.as_posix(),
                            video=str(video),
                            fold=fold,
                            frame=frame_index,
                            timestamp_ms=frame_index / fps * 1000,
                            bbox=[left, top, right, bottom],
                            yolo_conf=float(box.conf.item()),
                            teacher_score=probability,
                            suggestion=group,
                            label=None,
                            reviewed=False,
                            sharpness=float(
                                cv2.Laplacian(
                                    cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F
                                ).var()
                            ),
                            sha256=digest,
                        )
                        manifest.write(json.dumps(row, ensure_ascii=False) + "\n")
                        counts[group] += 1
                print(f"Finished {video.name}: {counts}", flush=True)
            finally:
                capture.release()
    (output / "summary.json").write_text(json.dumps(counts, indent=2), encoding="utf-8")


def main():
    import hydra

    @hydra.main(version_base=None, config_path="../configs", config_name="crops")
    def entry(cfg):
        run(cfg)

    entry()


if __name__ == "__main__":
    main()
