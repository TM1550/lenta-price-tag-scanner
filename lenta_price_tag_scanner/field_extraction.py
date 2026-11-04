"""Conservative extraction, provenance-aware aggregation and field coverage."""

import re
from collections import Counter
from datetime import datetime

import cv2
import numpy as np

from .constants import EXTRACTION_DEFAULTS
from .cpu_ocr import FIELDS, parse_text, prominent_price, qr_fields, valid_gtin

META_FIELDS = {"filename", "frame_timestamp", "x_min", "y_min", "x_max", "y_max"}
CONTENT_FIELDS = [field for field in FIELDS if field not in META_FIELDS]
UNKNOWN = {"", "нет", "null", "none", "nan", "unknown", "n/a"}


def is_filled(value):
    return value is not None and str(value).strip().lower() not in UNKNOWN


def coverage(rows, fields=CONTENT_FIELDS):
    counts = {field: sum(is_filled(row.get(field)) for row in rows) for field in fields}
    numerator = sum(counts.values())
    denominator = len(rows) * len(fields)
    return dict(
        rows=len(rows),
        fields=list(fields),
        filled_cells=numerator,
        possible_cells=denominator,
        coverage=numerator / denominator if denominator else 0,
        mean_filled_fields=numerator / len(rows) if rows else 0,
        per_field={
            key: dict(count=value, fraction=value / len(rows) if rows else 0)
            for key, value in counts.items()
        },
        note="Non-empty content fields, not correctness; metadata and absence markers excluded.",
    )


def text_fields(text):
    fields = {}
    # A label is required for semantic price assignment from unpositioned text.
    for field, label in [
        ("price_card", r"(?:по|с)\s+карт\w*"),
        ("price_default", r"без\s+карт\w*"),
        ("price_discount", r"цена\s+по\s+акции"),
    ]:
        found = re.search(label + r"\s*[:\-]?\s*(\d{1,6}[.,]\d{2})", text, re.I)
        if found:
            fields[field] = found.group(1).replace(",", ".")
    discount = set(re.findall(r"-?\s*\d{1,2}\s*%", text))
    if len(discount) == 1:
        fields["discount_amount"] = re.sub(r"\s", "", next(iter(discount)))
    date = re.search(r"\b\d{2}\.\d{2}\.\d{4}\s+\d{1,2}:\d{2}\b", text)
    if date:
        try:
            datetime.strptime(date.group(), "%d.%m.%Y %H:%M")
            fields["print_datetime"] = date.group()
        except ValueError:
            pass
    for field, label in [("id_sku", r"(?:артикул|арт\.?|sku)"), ("code", r"(?:код\s+зоны|зона)")]:
        found = re.search(label + r"\s*[:№]?\s*(\d{3,14})\b", text, re.I)
        if found:
            fields[field] = found.group(1)
    codes = {value for value in re.findall(r"(?<!\d)\d{8,14}(?!\d)", text) if valid_gtin(value)}
    if len(codes) == 1:
        fields["barcode"] = next(iter(codes))
    return fields


def crop_fields(
    image, text, lines, line_size, price_field="price_card", confidence=0.5, options=None
):
    options = {**EXTRACTION_DEFAULTS, **(options or {})}
    fields = text_fields(text)
    provenance = {key: "ocr_text" for key in fields}
    if lines:
        width, height = line_size
        positioned, codes = parse_text(lines, height, options["name_fraction"], confidence)
        name = positioned.get("product_name", "").strip()
        if name:
            fields["product_name"] = name
            provenance["product_name"] = "ocr_boxes"
        price = prominent_price(lines, height, width, confidence, options.get("price_layout"))
        if price and price_field and not fields.get(price_field):
            fields[price_field] = price
            provenance[price_field] = "price_template_heuristic"
    else:
        name_lines = []
        for line in text.splitlines():
            if re.search(r"[a-zA-Zа-яА-ЯёЁ]{3}", line) and not re.search(
                r"карта|картой|скидк|акци|руб|цена|\d{2}\.\d{2}\.\d{4}", line, re.I
            ):
                name_lines.append(line)
            elif name_lines:
                break
        if name_lines:
            fields["product_name"] = " ".join(name_lines[: options["max_name_lines"]])
            provenance["product_name"] = "vlm_text"
    pixels = np.asarray(image.convert("RGB"))
    hsv = cv2.cvtColor(pixels, cv2.COLOR_RGB2HSV)
    colored = (hsv[:, :, 1] > options["saturation_threshold"]) & (
        hsv[:, :, 2] > options["brightness_threshold"]
    )
    red = float(
        (
            colored
            & ((hsv[:, :, 0] < options["red_hue_low"]) | (hsv[:, :, 0] > options["red_hue_high"]))
        ).mean()
    )
    yellow = float(
        (
            colored
            & (hsv[:, :, 0] >= options["red_hue_low"])
            & (hsv[:, :, 0] < options["yellow_hue_high"])
        ).mean()
    )
    if max(red, yellow) > options["min_color_fraction"]:
        fields["color"] = "red" if red > yellow else "yellow"
        provenance["color"] = "pixel_color"
    import zxingcpp

    decoded = zxingcpp.read_barcodes(pixels)
    barcode_values, qr_values = set(), {}
    conflicts = set()
    for code in decoded:
        if str(code.format) in {"EAN13", "EAN8", "UPCA", "UPCE"} and valid_gtin(code.text):
            barcode_values.add(code.text)
        for key, value in qr_fields(code.text).items():
            if key in qr_values and qr_values[key] != value:
                conflicts.add(key)
            qr_values[key] = value
    for key in conflicts:
        qr_values.pop(key, None)
    for key, value in qr_values.items():
        if key == "qr_code_barcode" and not valid_gtin(value):
            continue
        if "price" in key and not re.fullmatch(r"\d{1,7}(?:[.,]\d{1,2})?", value):
            continue
        if "count" in key and not re.fullmatch(r"\d{1,6}", value):
            continue
        fields[key] = value
        provenance[key] = "decoded_qr"
    if valid_gtin(fields.get("qr_code_barcode", "")):
        barcode_values.add(fields["qr_code_barcode"])
    if len(barcode_values) == 1:
        fields["barcode"] = next(iter(barcode_values))
        provenance["barcode"] = "decoded_barcode"
    return fields, provenance


def fuse(observations):
    """Unique repeated observations vote; decoded symbols outrank optical guesses."""
    fields, provenance, conflicts = {}, {}, {}
    for field in CONTENT_FIELDS:
        available = [row for row in observations if is_filled(row["fields"].get(field))]
        if not available:
            continue
        decoded = [
            row for row in available if row["provenance"].get(field, "").startswith("decoded")
        ]
        available = decoded or available
        counts = Counter(str(row["fields"][field]).strip() for row in available)
        winners = counts.most_common()
        if len(winners) > 1 and winners[0][1] == winners[1][1]:
            conflicts[field] = dict(counts)
            # Names differ through OCR noise; retain best-quality observation, flagged for review.
            if field != "product_name":
                continue
        value = str(available[0]["fields"][field]) if field == "product_name" else winners[0][0]
        fields[field] = value
        provenance[field] = [
            row["image"] for row in available if str(row["fields"][field]) == value
        ]
    return fields, provenance, conflicts


def compatible_observations(observations):
    """Never build a synthetic tag by combining unrelated products after an ID switch."""
    from rapidfuzz.fuzz import ratio

    if not observations:
        return [], []
    selected, rejected = [observations[0]], []
    anchor = observations[0]["fields"]
    for row in observations[1:]:
        fields = row["fields"]
        anchor_code, candidate_code = anchor.get("barcode"), fields.get("barcode")
        same_code = bool(anchor_code and candidate_code and anchor_code == candidate_code)
        conflicting_code = bool(anchor_code and candidate_code and anchor_code != candidate_code)
        left, right = anchor.get("product_name", ""), fields.get("product_name", "")
        same_name = min(len(left), len(right)) >= 8 and ratio(left.lower(), right.lower()) >= 80
        if not conflicting_code and (same_code or same_name):
            selected.append(row)
        else:
            rejected.append(row["image"])
    return selected, rejected


def main(csv_path, output):
    import csv
    import json
    from pathlib import Path

    csv_path, output = Path(csv_path), Path(output)
    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != FIELDS:
            raise ValueError("Expected the 29-column Lenta schema")
        rows = list(reader)
    result = coverage(rows)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {key: result[key] for key in ("rows", "filled_cells", "possible_cells", "coverage")}
        )
    )


if __name__ == "__main__":
    import fire

    fire.Fire(main)
