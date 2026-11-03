"""Build portable, video-grouped pseudo-labelled datasets."""

import hashlib
import json
import shutil
from pathlib import Path

import hydra
from omegaconf import OmegaConf


def resolve(value):
    path = Path(value)
    return path.expanduser().resolve()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def checksum(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@hydra.main(version_base=None, config_path="../configs", config_name="build_datasets")
def main(cfg):
    import cv2
    from ultralytics import YOLO

    output = resolve(cfg.output)
    if output.exists():
        raise FileExistsError(output)
    candidates = resolve(cfg.candidates)
    videos = sorted(resolve(cfg.videos).glob("*.mp4"))
    if len(videos) != 5:
        raise ValueError("Expected exactly five source videos")
    if cfg.stride < 1 or not 0 <= cfg.candidate_conf <= cfg.label_conf <= 1:
        raise ValueError("Invalid sampling/confidence parameters")
    output.mkdir(parents=True)
    OmegaConf.save(cfg, output / "build_config.yaml")
    groups = {video.name: index for index, video in enumerate(videos)}
    classifier_rows, detector_rows = [], []
    for line in (candidates / "manifest.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["suggestion"] == "review":
            continue
        video_name = Path(row["video"]).name
        label = int(row["suggestion"] == "positive")
        relative = Path("efficient/images") / str(label) / video_name / Path(row["crop"]).name
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidates / row["crop"], target)
        classifier_rows.append(
            dict(
                image=relative.as_posix(),
                label=label,
                video=video_name,
                fold=groups[video_name],
                source="teacher_pseudo",
                reviewed=False,
                teacher_score=row["teacher_score"],
            )
        )
    detector = YOLO(str(resolve(cfg.weights)))
    skipped = 0
    for video in videos:
        capture = cv2.VideoCapture(str(video))
        if not capture.isOpened():
            raise RuntimeError(f"Cannot read {video}")
        frame_index = -1
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                frame_index += 1
                if frame_index % cfg.stride:
                    continue
                boxes = detector.predict(
                    frame,
                    conf=cfg.candidate_conf,
                    iou=cfg.iou,
                    imgsz=cfg.imgsz,
                    device=cfg.device,
                    verbose=False,
                )[0].boxes
                # Do not turn discarded uncertain boxes into background labels.
                if len(boxes) == 0 or any(float(box.conf.item()) < cfg.label_conf for box in boxes):
                    skipped += 1
                    continue
                relative = Path("yolo/images") / video.stem / f"{frame_index:08d}.jpg"
                label_path = output / "yolo/labels" / video.stem / f"{frame_index:08d}.txt"
                (output / relative).parent.mkdir(parents=True, exist_ok=True)
                label_path.parent.mkdir(parents=True, exist_ok=True)
                success, encoded = cv2.imencode(".jpg", frame)
                if not success:
                    raise RuntimeError("JPEG encoding failed")
                encoded.tofile(str(output / relative))
                labels = []
                for box in boxes:
                    center_x, center_y, width, height = box.xywhn[0].cpu().tolist()
                    labels.append(f"0 {center_x:.8f} {center_y:.8f} {width:.8f} {height:.8f}")
                label_path.write_text("\n".join(labels) + "\n", encoding="utf-8")
                detector_rows.append(
                    dict(
                        image=relative.as_posix(),
                        video=video.name,
                        fold=groups[video.name],
                        frame=frame_index,
                        boxes=len(labels),
                        source="teacher_pseudo",
                    )
                )
        finally:
            capture.release()
        print(f"Prepared {video.name}: {len(detector_rows)} frames total", flush=True)
    for name, rows in [("efficient", classifier_rows), ("yolo", detector_rows)]:
        write_json(output / name / "manifest.json", rows)
        folds = []
        for fold in range(5):
            train = [row["image"] for row in rows if row["fold"] != fold]
            valid = [row["image"] for row in rows if row["fold"] == fold]
            if not train or not valid:
                raise ValueError(f"Empty {name} partition in fold {fold}")
            folds.append(dict(fold=fold, train=train, val=valid))
        write_json(output / name / "folds.json", folds)
    write_json(
        output / "provenance.json",
        dict(
            videos={video.name: checksum(video) for video in videos},
            detector_sha256=checksum(resolve(cfg.weights)),
            candidates_manifest_sha256=checksum(candidates / "manifest.jsonl"),
            groups=groups,
            skipped_yolo_frames=skipped,
            caveat="Teacher agreement only, not independent ground-truth evaluation",
        ),
    )
    print(f"Completed: {len(classifier_rows)} crops, {len(detector_rows)} YOLO frames")


if __name__ == "__main__":
    main()
