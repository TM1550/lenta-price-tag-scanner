"""Reconstruct CPU-only fields from the same recorded OCR observations and tracks."""

import csv
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import fire

from lenta_price_tag_scanner.cpu_ocr import FIELDS, valid_gtin
from lenta_price_tag_scanner.field_extraction import (
    META_FIELDS,
    compatible_observations,
    coverage,
    fuse,
)


def main(result, catalogue_db):
    args = SimpleNamespace(result=Path(result), catalogue_db=Path(catalogue_db))
    connection = sqlite3.connect(args.catalogue_db.resolve().as_uri() + "?mode=ro", uri=True)
    rows = []
    for line in (args.result / "details.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        compatible, _ = compatible_observations(record["observations"])
        fields, _, _ = fuse(compatible)
        codes = {
            fields[key] for key in ("barcode", "qr_code_barcode") if valid_gtin(fields.get(key, ""))
        }
        matches = set()
        for code in codes:
            matches.update(
                connection.execute(
                    "SELECT DISTINCT fullname, code FROM products WHERE code=?", (code,)
                ).fetchall()
            )
        if len(matches) == 1:
            fields["product_name"] = next(iter(matches))[0]
        row = dict.fromkeys(FIELDS, "")
        row.update(fields)
        row.update({key: record["fields"][key] for key in META_FIELDS})
        rows.append(row)
    connection.close()
    with (args.result / "baseline_same_tracks.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    baseline = coverage(rows)
    final = json.loads((args.result / "metrics.json").read_text(encoding="utf-8"))
    result = dict(
        baseline=baseline,
        final={key: final[key] for key in baseline},
        coverage_delta=final["coverage"] - baseline["coverage"],
        note="Same tracks/crops; reconstructed without VLM. Coverage difference is not accuracy gain.",
    )
    (args.result / "vlm_ablation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        json.dumps(
            dict(
                baseline=baseline["coverage"],
                final=final["coverage"],
                baseline_filled=baseline["filled_cells"],
                final_filled=final["filled_cells"],
            )
        )
    )


if __name__ == "__main__":
    fire.Fire(main)
