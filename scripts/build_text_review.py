"""Create a local review page and comparisons from completed OCR benchmark runs."""

import html
import json
import os
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import quote

import fire


def main(results, images):
    args = SimpleNamespace(results=Path(results), images=Path(images))
    root = args.results.resolve()
    exploratory = {
        json.loads(line)["image"]
        for line in (root / "ocr_qwen3_bounded/predictions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    }
    comparison = {}
    by_backend = {}
    for backend, folder in [("easyocr", "ocr_benchmark_easyocr"), ("qwen3", "ocr_qwen3_full")]:
        rows = json.loads((root / folder / "evaluation.json").read_text(encoding="utf-8"))
        by_backend[backend] = {row["image"]: row for row in rows}
        comparison[backend] = {}
        for subset, selected in [
            ("all", rows),
            ("excluding_5_exploratory", [row for row in rows if row["image"] not in exploratory]),
        ]:
            comparison[backend][subset] = {
                "images": len(selected),
                "cer": sum(row["char_errors"] for row in selected)
                / sum(row["chars"] for row in selected),
                "wer": sum(row["word_errors"] for row in selected)
                / sum(row["words"] for row in selected),
            }
    (root / "comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    parts = [
        '<!doctype html><html lang="ru"><meta charset="utf-8"><title>OCR: проверка</title>',
        "<style>body{font:16px system-ui;margin:24px;background:#f5f6f8;color:#17212b}"
        "section{display:grid;grid-template-columns:300px 1fr 1fr 1fr;gap:20px;"
        "background:white;margin:16px 0;padding:16px;border-radius:8px}"
        ".image{width:300px;height:300px;display:flex;align-items:center;justify-content:center}"
        "img{max-width:270px;max-height:270px;transform:rotate(-90deg)}"
        "p{white-space:pre-wrap;overflow-wrap:anywhere}small{color:#546477}"
        "@media(max-width:900px){section{grid-template-columns:1fr 1fr}}</style>",
        "<h1>Сравнение распознавания ценников</h1>",
        "<p>294 кропа. Текст нормализован для CER/WER. Части эталона могут отсутствовать "
        "на обрезанном изображении. Предсказания требуют проверки. Пять exploratory "
        "примеров использовались при настройке запроса; модель не дообучалась.</p>",
        "<pre>" + html.escape(json.dumps(comparison, indent=2)) + "</pre>",
    ]
    for name, row in by_backend["qwen3"].items():
        relative = os.path.relpath(args.images.resolve() / name, root).replace(os.sep, "/")
        source = quote(relative, safe="/.")
        parts.append(
            '<section><div><div class="image"><img loading="lazy" src="'
            + source
            + '" alt="Кроп ценника"></div><small>'
            + html.escape(name)
            + (" · exploratory" if name in exploratory else "")
            + "</small></div>"
        )
        for title, text in [
            ("Эталон", row["reference"]),
            ("Qwen3-VL", row["prediction"]),
            ("EasyOCR", by_backend["easyocr"][name]["prediction"]),
        ]:
            parts.append("<div><b>" + title + "</b><p>" + html.escape(text) + "</p></div>")
        parts.append("</section>")
    parts.append("</html>")
    (root / "review.html").write_text("\n".join(parts), encoding="utf-8")
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    fire.Fire(main)
