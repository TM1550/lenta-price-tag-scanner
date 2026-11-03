"""Build a portable HTML inspection page next to the final CSV and crop images."""

import html
import json
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import quote

import fire


def main(result):
    args = SimpleNamespace(result=Path(result))
    root = args.result
    metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    parts = [
        '<!doctype html><html lang="ru"><meta charset="utf-8"><title>Видео → CSV</title>',
        "<style>body{font:16px system-ui;margin:24px;background:#f5f6f8}"
        "section{background:white;padding:20px;margin:16px 0;display:flex;gap:30px}"
        ".crop{width:300px;height:320px;display:flex;align-items:center;justify-content:center}"
        "img{max-width:280px;max-height:280px;transform:rotate(-90deg)}"
        "td{padding:4px 12px;vertical-align:top}pre{white-space:pre-wrap}"
        "@media(max-width:700px){section{display:block}}</style>",
        "<h1>Видео → ценники → CSV</h1><p>Одна строка на сегмент трека. Возможны "
        "дубли физических ценников. Заполненность не означает правильность.</p>",
        f'<p>Строк: {metrics["rows"]}; заполненность: {metrics["coverage"]:.2%}; '
        f'полей в среднем: {metrics["mean_filled_fields"]:.2f} из 23.</p>',
        '<p><a href="submission.csv">CSV</a> · <a href="metrics.json">Метрики</a></p>',
    ]
    for line in (root / "details.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        source = quote(row["observations"][0]["image"], safe="/.")
        parts.append(
            '<section><div><div class="crop"><img loading="lazy" src="' + source + '"></div>'
        )
        parts.append(
            html.escape(row["filename"]) + " · track " + str(row["track_id"]) + "</div><div><table>"
        )
        for key, value in row["fields"].items():
            if value != "":
                parts.append(
                    "<tr><td>"
                    + html.escape(key)
                    + "</td><td>"
                    + html.escape(str(value))
                    + "</td></tr>"
                )
        parts.append(
            "</table><details><summary>Исходный текст и происхождение</summary><pre>"
            + html.escape(json.dumps(row, ensure_ascii=False, indent=2))
            + "</pre></details></div></section>"
        )
    parts.append("</html>")
    (root / "review.html").write_text("\n".join(parts), encoding="utf-8")


if __name__ == "__main__":
    fire.Fire(main)
