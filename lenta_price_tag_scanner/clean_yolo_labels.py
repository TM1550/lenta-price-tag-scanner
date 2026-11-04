"""Convert mixed Roboflow labels to detection boxes, then filter and apply NMS."""

import hashlib
import json
import math
import re
import shutil
from collections import Counter
from pathlib import Path

import hydra
from omegaconf import OmegaConf
from PIL import Image, ImageDraw


def checksum(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_label(line, line_number, class_count):
    tokens = line.split()
    values = [float(value) for value in tokens]
    if len(values) < 5 or any(not math.isfinite(value) for value in values):
        raise ValueError(f"Invalid label at line {line_number}")
    cls = values[0]
    if cls != int(cls) or not 0 <= cls < class_count:
        raise ValueError(f"Invalid class at line {line_number}")
    coords = values[1:]
    if any(value < -1e-7 or value > 1 + 1e-7 for value in coords):
        raise ValueError(f"Non-normalized coordinates at line {line_number}")
    if len(coords) == 4:
        center_x, center_y, width, height = coords
        if width < 0 or height < 0:
            raise ValueError(f"Negative box size at line {line_number}")
        box = [
            center_x - width / 2,
            center_y - height / 2,
            center_x + width / 2,
            center_y + height / 2,
        ]
        kind = "bbox"
    elif len(coords) >= 6 and len(coords) % 2 == 0:
        box = [min(coords[::2]), min(coords[1::2]), max(coords[::2]), max(coords[1::2])]
        kind = "polygon"
    else:
        raise ValueError(f"Expected xywh or polygon at line {line_number}")
    clipped = [max(0.0, min(1.0, value)) for value in box]
    return dict(line=line_number, cls=int(cls), xyxy=clipped, kind=kind, clipped=clipped != box)


def area(box):
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def iou(first, second):
    intersection = area(
        [
            max(first[0], second[0]),
            max(first[1], second[1]),
            min(first[2], second[2]),
            min(first[3], second[3]),
        ]
    )
    union = area(first) + area(second) - intersection
    return intersection / union if union > 0 else 0.0


def clean_boxes(boxes, width, height, min_side_px, iou_threshold):
    """Class-aware greedy NMS; absent confidences are replaced by box area."""
    if width <= 0 or height <= 0 or min_side_px < 0 or not 0 <= iou_threshold <= 1:
        raise ValueError("Invalid dimensions or thresholds")
    candidates, removed = [], []
    for box in boxes:
        x1, y1, x2, y2 = box["xyxy"]
        side_x, side_y = (x2 - x1) * width, (y2 - y1) * height
        if min(side_x, side_y) + 1e-6 < min_side_px or area(box["xyxy"]) <= 0:
            removed.append(dict(box, reason="small", width_px=side_x, height_px=side_y))
        else:
            candidates.append(box)
    kept = []
    for box in sorted(
        candidates, key=lambda candidate: (-area(candidate["xyxy"]), candidate["line"])
    ):
        matches = [(iou(box["xyxy"], k["xyxy"]), k) for k in kept if k["cls"] == box["cls"]]
        overlap, match = max(matches, key=lambda item: item[0], default=(0.0, None))
        if overlap > iou_threshold:
            removed.append(dict(box, reason="nms", kept_line=match["line"], iou=overlap))
        else:
            kept.append(box)
    return sorted(kept, key=lambda candidate: candidate["line"]), sorted(
        removed, key=lambda candidate: candidate["line"]
    )


def yolo_line(box):
    x1, y1, x2, y2 = box["xyxy"]
    return f"{box['cls']} {(x1 + x2) / 2:.10f} {(y1 + y2) / 2:.10f} {x2 - x1:.10f} {y2 - y1:.10f}"


def preview(image, boxes, removed, destination):
    with Image.open(image) as im:
        im = im.convert("RGB")
        width, height = im.size
        im.thumbnail((1280, 720))
        scale_x, scale_y = im.width / width, im.height / height
        draw = ImageDraw.Draw(im)
        for box in boxes + removed:
            reason = box.get("reason")
            color = {"nms": "red", "small": "yellow"}.get(reason, "lime")
            x1, y1, x2, y2 = box["xyxy"]
            rect = [
                x1 * width * scale_x,
                y1 * height * scale_y,
                x2 * width * scale_x,
                y2 * height * scale_y,
            ]
            draw.rectangle(rect, outline=color, width=2)
            draw.text((rect[0], rect[1]), f"{box['line']} {reason or 'keep'}", fill=color)
        im.save(destination, quality=90)


@hydra.main(version_base=None, config_path="../configs", config_name="clean_yolo_labels")
def main(cfg):
    source, output = Path(cfg.source).resolve(), Path(cfg.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Source and output must be separate directories")
    metadata = OmegaConf.to_container(OmegaConf.load(source / "data.yaml"), resolve=True)
    names = metadata["names"]
    if isinstance(names, dict):
        names = [names[key] for key in sorted(names, key=int)]
    if not names or metadata.get("nc", len(names)) != len(names):
        raise ValueError("Invalid class metadata")
    pattern = re.compile(r"^(?P<video>.+)__f(?P<frame>\d+)__t(?P<time>\d+)ms")
    source_manifest = Path(cfg.frame_manifest)
    original = json.loads(source_manifest.read_text(encoding="utf-8"))
    original = {Path(row["image"]).stem: row for row in original}
    tasks, seen = [], set()
    for split in ("train", "valid", "test"):
        image_dir = source / split / "images"
        images = sorted(path for path in image_dir.glob("*") if path.is_file())
        for image in images:
            if image.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                raise ValueError(f"Unsupported image: {image}")
            match = pattern.match(image.stem)
            if match is None or match[0] not in original:
                raise ValueError(f"Cannot match image to frame manifest: {image.name}")
            if match[0] in seen:
                raise ValueError(f"Duplicate source frame: {image.name}")
            seen.add(match[0])
            label = source / split / "labels" / (image.stem + ".txt")
            lines = label.read_text(encoding="utf-8-sig").splitlines()
            boxes = [
                parse_label(line, number, len(names))
                for number, line in enumerate(lines, 1)
                if line.strip()
            ]
            with Image.open(image) as im:
                width, height = im.size
                im.verify()
            provenance = original[match[0]]
            if (width, height) != (provenance["width"], provenance["height"]):
                raise ValueError(f"Roboflow changed image dimensions: {image.name}")
            kept, removed = clean_boxes(boxes, width, height, cfg.min_side_px, cfg.iou_threshold)
            tasks.append(
                dict(
                    image=image,
                    label=label,
                    split=split,
                    boxes=boxes,
                    kept=kept,
                    removed=removed,
                    provenance=provenance,
                )
            )
        expected = {image.stem for image in images}
        actual = {label.stem for label in (source / split / "labels").glob("*.txt")}
        if expected != actual:
            raise ValueError(f"Image/label mismatch in {split}")
    if not tasks:
        raise ValueError("No images found")
    output.mkdir(parents=True)
    OmegaConf.save(cfg, output / "clean_config.yaml")
    (output / "previews").mkdir()
    totals = Counter()
    by_video = {}
    manifest, audit = [], []
    for task in tasks:
        image, label = task["image"], task["label"]
        dest_image = output / task["split"] / "images" / image.name
        dest_label = output / task["split"] / "labels" / label.name
        dest_image.parent.mkdir(parents=True, exist_ok=True)
        dest_label.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image, dest_image)
        content = "\n".join(yolo_line(box) for box in task["kept"])
        dest_label.write_text(content + ("\n" if content else ""), encoding="utf-8")
        provenance = task["provenance"]
        record = dict(
            image=str(dest_image.relative_to(output)).replace("\\", "/"),
            label=str(dest_label.relative_to(output)).replace("\\", "/"),
            video=provenance["video"],
            fold=provenance["fold"],
            frame_index=provenance["frame_index"],
            timestamp_ms=provenance["timestamp_ms"],
            source_image_sha256=checksum(image),
            source_label_sha256=checksum(label),
            cleaned_label_sha256=checksum(dest_label),
            input_boxes=len(task["boxes"]),
            kept_boxes=len(task["kept"]),
            empty_after_filter=not task["kept"],
            source_was_empty=not task["boxes"],
        )
        manifest.append(record)
        changes = Counter(removed_box["reason"] for removed_box in task["removed"])
        count = Counter(
            images=1,
            input_boxes=len(task["boxes"]),
            kept_boxes=len(task["kept"]),
            removed_small=changes["small"],
            removed_nms=changes["nms"],
            polygon_input=sum(candidate["kind"] == "polygon" for candidate in task["boxes"]),
            bbox_input=sum(candidate["kind"] == "bbox" for candidate in task["boxes"]),
            clipped_input=sum(candidate["clipped"] for candidate in task["boxes"]),
            empty_input=int(not task["boxes"]),
            empty_output=int(not task["kept"]),
        )
        totals.update(count)
        by_video.setdefault(record["video"], Counter()).update(count)
        audit.append(dict(image=record["image"], kept=task["kept"], removed=task["removed"]))
    # Prioritize changes rather than the first frames of one video.
    changed = sorted(
        tasks, key=lambda task_item: (-len(task_item["removed"]), task_item["image"].name)
    )
    for task in changed[: cfg.preview_limit]:
        preview(
            task["image"],
            task["kept"],
            task["removed"],
            output / "previews" / (task["image"].stem + ".jpg"),
        )
    dataset = {
        split: f"{split}/images"
        for split in ("train", "valid", "test")
        if (output / split / "images").exists()
    }
    dataset.update(nc=len(names), names=names)
    # No invented validation split: OOF must be rebuilt from the manifest by source video.
    OmegaConf.save(OmegaConf.create(dataset), output / "data.yaml")
    for filename in ("README.dataset.txt", "README.roboflow.txt", "data.yaml"):
        shutil.copy2(source / filename, output / ("source_" + filename))
    missing = sorted(set(original) - seen)
    report = dict(
        totals=totals,
        by_video=by_video,
        missing_exported_frames=missing,
        iou_threshold=cfg.iou_threshold,
        min_side_px=cfg.min_side_px,
        nms_priority="descending clipped bbox area; source line tie-break",
        source=str(source),
        source_yaml_sha256=checksum(source / "data.yaml"),
        frame_manifest_sha256=checksum(source_manifest),
        note="Empty filtered frames require review; current split is not an OOF split.",
    )
    for name, value in (
        ("manifest.json", manifest),
        ("audit.json", audit),
        ("report.json", report),
    ):
        (output / name).write_text(
            json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    (output / "README.txt").write_text(
        "Detection labels only (class cx cy w h). Source files were preserved.\n"
        "Green=kept, red=NMS removal, yellow=small removal in previews/.\n"
        "review.html: all changed frames; source provenance in manifest.json.\n"
        "Empty labels are retained but require manual confirmation before training.\n"
        "Rebuild OOF by video using manifest.json; do not randomly split frames.\n",
        encoding="utf-8",
    )
    write_review(output, tasks)
    (output / "COMPLETE.txt").write_text("Label cleaning finished.\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def write_review(output, tasks):
    """Offline viewer overlays all changes on untouched full-resolution images."""
    data = []
    for task in tasks:
        if not task["removed"] and task["boxes"]:
            continue
        relative = task["image"].name
        data.append(
            dict(
                image=f"{task['split']}/images/{relative}",
                kept=task["kept"],
                removed=task["removed"],
            )
        )
    html = """<!doctype html><meta charset="utf-8"><title>YOLO label cleaning review</title>
<style>body{font:16px sans-serif;background:#222;color:white}canvas{width:100%;max-width:1600px}
button{font-size:18px;margin:8px}#status{overflow-wrap:anywhere}</style>
<p>Green: kept; red: NMS; yellow: too small. All changed or empty frames.</p>
<button onclick="show(-1)">Previous</button><button onclick="show(1)">Next</button>
<label><input id="before" type="checkbox" checked onchange="draw()">Show removed</label>
<p id="status"></p><canvas id="canvas"></canvas><pre id="details"></pre>
<script>const data=DATA;let pos=0;const img=new Image();
const canvas=document.getElementById('canvas'), ctx=canvas.getContext('2d');
function draw(){canvas.width=img.naturalWidth;canvas.height=img.naturalHeight;ctx.drawImage(img,0,0);
let row=data[pos];let boxes=document.getElementById('before').checked?row.kept.concat(row.removed):row.kept;
ctx.lineWidth=5;ctx.font='28px sans-serif';for(const b of boxes){let q=b.xyxy;
let x=q[0]*canvas.width,y=q[1]*canvas.height,w=(q[2]-q[0])*canvas.width,h=(q[3]-q[1])*canvas.height;
ctx.strokeStyle=b.reason==='nms'?'red':b.reason==='small'?'yellow':'lime';ctx.fillStyle=ctx.strokeStyle;
ctx.strokeRect(x,y,w,h);ctx.fillText(b.line+' '+(b.reason||'keep'),x,y+28);}}
function show(delta){if(!data.length){document.getElementById('status').textContent='No changes';return;}
pos=(pos+delta+data.length)%data.length;let row=data[pos];
document.getElementById('status').textContent=(pos+1)+' / '+data.length+' '+row.image;
document.getElementById('details').textContent=JSON.stringify(row.removed,null,2);img.src=row.image;}
img.onload=draw;show(0);</script>"""
    (output / "review.html").write_text(html.replace("DATA", json.dumps(data)), encoding="utf-8")


if __name__ == "__main__":
    main()
