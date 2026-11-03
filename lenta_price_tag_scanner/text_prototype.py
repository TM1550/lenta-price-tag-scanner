"""Local image-to-text prototype; reference labels are used only after inference."""

import json
import re
import time
import unicodedata
from collections import Counter

import hydra
from omegaconf import OmegaConf
from PIL import Image

from .build_datasets import checksum, resolve, write_json


def normalize(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).lower()).strip()


def adapt_paddle_mask(model):
    """Pinned upstream custom code uses the newer name of a Transformers 4.x argument."""
    import inspect
    import sys

    module = sys.modules[type(model).__module__]
    original = module.create_causal_mask
    parameters = inspect.signature(original).parameters
    if "inputs_embeds" not in parameters and "input_embeds" in parameters:

        def compatible(*args, **kwargs):
            if "inputs_embeds" in kwargs:
                kwargs["input_embeds"] = kwargs.pop("inputs_embeds")
            return original(*args, **kwargs)

        module.create_causal_mask = compatible


def ordered_text(lines):
    """Approximate row order from boxes rather than detector's arbitrary ordering."""
    boxes = []
    for bbox, text, confidence in lines:
        top = min(point[1] for point in bbox)
        bottom = max(point[1] for point in bbox)
        boxes.append(
            dict(text=text, top=top, height=bottom - top, left=min(point[0] for point in bbox))
        )
    ordered, current = [], []
    for box in sorted(boxes, key=lambda item: item["top"]):
        if (
            current
            and box["top"]
            > min(item["top"] for item in current) + min(item["height"] for item in current) * 0.6
        ):
            ordered.extend(sorted(current, key=lambda item: item["left"]))
            current = []
        current.append(box)
    ordered.extend(sorted(current, key=lambda item: item["left"]))
    return " ".join(item["text"] for item in ordered)


class TextReader:
    def __init__(self, cfg):
        import torch

        torch.set_num_threads(cfg.cpu_threads)
        self.cfg = cfg
        self.model = None
        self.processor = None
        self.last_lines = []
        self.last_size = None
        if cfg.backend == "easyocr":
            import easyocr

            root = resolve(cfg.easyocr_models)
            self.model = easyocr.Reader(
                ["ru", "en"],
                gpu=False if cfg.device == "cpu" else cfg.device,
                model_storage_directory=str(root),
                user_network_directory=str(root / "user_network"),
                verbose=False,
                quantize=cfg.device == "cpu",
            )
        elif cfg.backend == "paddle":
            import os

            os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(resolve(cfg.paddle_cache)))
            from paddleocr import PaddleOCR

            local_models = {}
            for option in ("text_detection_model_dir", "text_recognition_model_dir"):
                configured = cfg.get(option)
                if configured:
                    directory = resolve(configured)
                    if not directory.is_dir():
                        raise FileNotFoundError(directory)
                    local_models[option] = str(directory)
            self.model = PaddleOCR(
                text_detection_model_name="PP-OCRv5_mobile_det",
                text_recognition_model_name=cfg.paddle_rec,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                device="cpu",
                enable_mkldnn=False,
                cpu_threads=cfg.cpu_threads,
                **local_models,
            )
        elif cfg.backend in ("qwen", "paddlevl"):
            from transformers import (
                AutoModelForCausalLM,
                AutoModelForImageTextToText,
                AutoProcessor,
            )

            model_path = cfg.paddle_vl_model if cfg.backend == "paddlevl" else cfg.vlm_model
            extra = (
                {"trust_remote_code": True, "revision": cfg.paddle_vl_revision}
                if cfg.backend == "paddlevl"
                else {"revision": cfg.vlm_revision}
            )
            extra["local_files_only"] = cfg.get("local_files_only", False)
            self.processor = AutoProcessor.from_pretrained(
                model_path, min_pixels=cfg.min_pixels, max_pixels=cfg.max_pixels, **extra
            )
            model_class = (
                AutoModelForCausalLM if cfg.backend == "paddlevl" else AutoModelForImageTextToText
            )
            self.model = (
                model_class.from_pretrained(
                    model_path,
                    **extra,
                    torch_dtype=torch.bfloat16 if cfg.device != "cpu" else torch.float32,
                    attn_implementation="sdpa",
                )
                .to(cfg.device)
                .eval()
            )
            if cfg.backend == "paddlevl":
                adapt_paddle_mask(self.model)
        else:
            raise ValueError("backend must be easyocr, paddle, qwen or paddlevl")

    def read(self, image):
        from .crop_quality import preprocess

        image = preprocess(image, self.cfg.get("preprocess", "none"))
        if self.cfg.backend in ("qwen", "paddlevl") and self.cfg.get("zones", False):
            middle = image.height // 2
            upper, cut_upper = self._read(image.crop((0, 0, image.width, middle)))
            lower, cut_lower = self._read(image.crop((0, middle, image.width, image.height)))
            self.last_size = image.size
            return upper + "\n" + lower, cut_upper or cut_lower
        return self._read(image)

    def _read(self, image):
        import torch

        self.last_size = image.size
        self.last_lines = []

        if self.cfg.backend in ("easyocr", "paddle"):
            import numpy as np

            scale = min(self.cfg.scale, self.cfg.max_side / max(image.size))
            resized = image.resize(
                (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
            )
            self.last_size = resized.size
            if self.cfg.backend == "paddle":
                result = next(iter(self.model.predict(np.asarray(resized))))
                lines = [
                    (box.tolist(), str(text), float(score))
                    for box, text, score in zip(
                        result["rec_polys"], result["rec_texts"], result["rec_scores"]
                    )
                ]
                self.last_lines = lines
                return ordered_text(lines), False
            lines = self.model.readtext(
                np.asarray(resized),
                detail=1,
                paragraph=False,
                decoder="greedy",
                batch_size=1,
                workers=0,
            )
            self.last_lines = lines
            return ordered_text(lines), False
        messages = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": self.cfg.prompt}],
            }
        ]
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=[prompt], images=[image], return_tensors="pt").to(
            self.cfg.device
        )
        if self.cfg.backend == "paddlevl":
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": "OCR:"},
                    ],
                }
            ]
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.cfg.device)
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=self.cfg.max_new_tokens,
                repetition_penalty=1.05,
                no_repeat_ngram_size=6,
            )
        tokens = generated[:, inputs.input_ids.shape[1] :]
        text = self.processor.batch_decode(
            tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return text.strip(), tokens.shape[1] >= self.cfg.max_new_tokens


@hydra.main(version_base=None, config_path="../configs", config_name="text_prototype")
def main(cfg):
    import torch
    from rapidfuzz.distance import Levenshtein

    metadata = {}
    if cfg.crop_manifest:
        manifest = resolve(cfg.crop_manifest)
        files = []
        for line in manifest.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            path = (manifest.parent / record["image"]).resolve()
            path.relative_to(manifest.parent.resolve())
            if path in metadata:
                raise ValueError("Duplicate crop path in manifest")
            files.append(path)
            metadata[path] = record
    else:
        source = resolve(cfg.source)
        files = (
            [source]
            if source.is_file()
            else sorted(
                path
                for path in source.iterdir()
                if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
            )
        )
    if not files:
        raise ValueError("No input images")
    if cfg.rotation_ccw not in [0, 90, 180, 270] or cfg.cpu_threads < 1 or cfg.per_video < 0:
        raise ValueError("Invalid rotation, threads or sample count")
    groups = Counter()
    selected = []
    for path in files:
        video = path.stem.split("_track")[0] if "_track" in path.stem else "input"
        if path in metadata:
            video = metadata[path]["filename"]
        if cfg.per_video and groups[video] >= cfg.per_video:
            continue
        groups[video] += 1
        selected.append((path, video))
    output = resolve(cfg.output)
    output.mkdir(parents=True, exist_ok=False)
    OmegaConf.save(cfg, output / "config.yaml")
    if cfg.device != "cpu":
        torch.cuda.reset_peak_memory_stats()
    reader = TextReader(cfg)
    evaluated = []
    elapsed = 0
    with (output / "predictions.jsonl").open("w", encoding="utf-8") as stream:
        for path, video in selected:
            tick = time.perf_counter()
            with Image.open(path) as original:
                image = original.convert("RGB").rotate(cfg.rotation_ccw, expand=True)
            text, truncated = reader.read(image)
            seconds = time.perf_counter() - tick
            elapsed += seconds
            row = dict(
                image=path.name,
                video=video,
                text=text,
                seconds=seconds,
                truncated=truncated,
                image_sha256=checksum(path),
                backend=cfg.backend,
            )
            if path in metadata:
                row["source_observation"] = metadata[path]
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            if cfg.labels:
                label_path = resolve(cfg.labels) / (path.stem + ".txt")
                if label_path.is_file():
                    reference = normalize(label_path.read_text(encoding="utf-8-sig"))
                    prediction = normalize(text)
                    if reference:
                        evaluated.append(
                            dict(
                                image=path.name,
                                video=video,
                                reference=reference,
                                prediction=prediction,
                                char_errors=Levenshtein.distance(reference, prediction),
                                chars=len(reference),
                                word_errors=Levenshtein.distance(
                                    reference.split(), prediction.split()
                                ),
                                words=len(reference.split()),
                            )
                        )
            print(
                f"{len(evaluated) if cfg.labels else path.name}: {video} {seconds:.2f}s", flush=True
            )
    summary = dict(
        images=len(selected),
        by_video=dict(groups),
        backend=cfg.backend,
        device=cfg.device,
        processing_seconds=elapsed,
        peak_gpu_bytes=torch.cuda.max_memory_allocated() if cfg.device != "cpu" else 0,
    )
    if evaluated:
        summary["evaluated"] = len(evaluated)
        summary["cer"] = sum(row["char_errors"] for row in evaluated) / sum(
            row["chars"] for row in evaluated
        )
        summary["wer"] = sum(row["word_errors"] for row in evaluated) / sum(
            row["words"] for row in evaluated
        )
        summary["per_video"] = {
            video: dict(
                cer=sum(row["char_errors"] for row in evaluated if row["video"] == video)
                / sum(row["chars"] for row in evaluated if row["video"] == video),
                count=sum(row["video"] == video for row in evaluated),
            )
            for video in sorted({row["video"] for row in evaluated})
        }
        write_json(output / "evaluation.json", evaluated)
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
