"""CPU text extraction and conservative local catalogue lookup."""

import csv
import json
import re
import sqlite3
import time
from collections import Counter
from urllib.parse import parse_qsl, urlparse

import hydra
from omegaconf import OmegaConf

from .build_datasets import checksum, resolve, write_json

FIELDS = (
    "filename product_name price_default price_card price_discount barcode discount_amount "
    "id_sku print_datetime code additional_info color special_symbols frame_timestamp "
    "x_min y_min x_max y_max qr_code_barcode price1_qr price2_qr price3_qr price4_qr "
    "wholesale_level_1_count wholesale_level_1_price wholesale_level_2_count "
    "wholesale_level_2_price action_price_qr action_code_qr"
).split()
QR_KEYS = {
    "barcode": "qr_code_barcode",
    "b": "qr_code_barcode",
    "price1": "price1_qr",
    "p1": "price1_qr",
    "price2": "price2_qr",
    "p2": "price2_qr",
    "price3": "price3_qr",
    "p3": "price3_qr",
    "price4": "price4_qr",
    "p4": "price4_qr",
    "wholesaleLevel1Count": "wholesale_level_1_count",
    "wL1C": "wholesale_level_1_count",
    "wholesaleLevel1Price": "wholesale_level_1_price",
    "wL1P": "wholesale_level_1_price",
    "wholesaleLevel2Count": "wholesale_level_2_count",
    "wL2C": "wholesale_level_2_count",
    "wholesaleLevel2Price": "wholesale_level_2_price",
    "wL2P": "wholesale_level_2_price",
    "actionPrice": "action_price_qr",
    "aP": "action_price_qr",
    "actionCode": "action_code_qr",
    "aC": "action_code_qr",
}


def valid_gtin(value):
    if not value.isdigit() or len(value) not in (8, 12, 13, 14):
        return False
    total = sum(
        int(digit) * (3 if index % 2 == 0 else 1)
        for index, digit in enumerate(reversed(value[:-1]))
    )
    return (10 - total % 10) % 10 == int(value[-1])


def qr_fields(payload):
    try:
        decoded = json.loads(payload)
    except (ValueError, TypeError):
        decoded = dict(parse_qsl(urlparse(payload).query or payload, keep_blank_values=True))
    if not isinstance(decoded, dict):
        return {}
    return {
        QR_KEYS[key]: str(value)
        for key, value in decoded.items()
        if key in QR_KEYS and value is not None and not isinstance(value, (dict, list))
    }


def catalogue(csv_path, db_path, encoding):
    """Build once, retain ambiguous codes, reject stale or partial indexes."""
    signature = checksum(csv_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    connection.execute("CREATE TABLE IF NOT EXISTS metadata (hash TEXT)")
    previous = connection.execute("SELECT hash FROM metadata").fetchone()
    if previous:
        if previous[0] != signature:
            connection.close()
            raise ValueError("Catalogue changed; choose a new catalogue_db path")
        return connection
    with connection:
        connection.execute("DROP TABLE IF EXISTS products")
        connection.execute("DROP TABLE IF EXISTS names")
        connection.execute("CREATE TABLE products (fullname TEXT, code TEXT)")
        connection.execute("CREATE VIRTUAL TABLE names USING fts5(fullname, code UNINDEXED)")
        with csv_path.open(encoding=encoding, newline="") as stream:
            reader = csv.DictReader(stream, delimiter=";")
            if reader.fieldnames != ["fullname", "code"]:
                raise ValueError(f"Unexpected catalogue columns: {reader.fieldnames}")
            batch = []
            for row in reader:
                batch.append((row["fullname"].strip(), row["code"].strip()))
                if len(batch) == 5000:
                    connection.executemany("INSERT INTO products VALUES (?,?)", batch)
                    connection.executemany("INSERT INTO names VALUES (?,?)", batch)
                    batch.clear()
            connection.executemany("INSERT INTO products VALUES (?,?)", batch)
            connection.executemany("INSERT INTO names VALUES (?,?)", batch)
        connection.execute("CREATE INDEX by_code ON products(code)")
        connection.execute("INSERT INTO metadata VALUES (?)", (signature,))
    return connection


def match_catalog(connection, codes, name, top_k):
    matches = set()
    for code in codes:
        matches.update(
            connection.execute(
                "SELECT DISTINCT fullname, code FROM products WHERE code=?", (code,)
            ).fetchall()
        )
    if len(matches) == 1:
        fullname, code = next(iter(matches))
        return dict(fullname=fullname, code=code), []
    tokens = [token for token in re.findall(r"[^\W_]+", name.lower()) if len(token) >= 3][:8]
    if not tokens:
        return None, []
    query = " OR ".join('"' + token + '"' for token in tokens)
    candidates = connection.execute(
        "SELECT fullname, code, min(rank) AS rank FROM names WHERE names MATCH ? GROUP BY fullname, code ORDER BY rank LIMIT ?",
        (query, top_k),
    ).fetchall()
    return None, [
        dict(fullname=fullname, code=code, rank=rank) for fullname, code, rank in candidates
    ]


def parse_text(lines, height, name_fraction, confidence):
    usable = [(bbox, text, score) for bbox, text, score in lines if score >= confidence]
    text = " ".join(item[1] for item in usable)
    names = [
        item[1]
        for item in sorted(usable, key=lambda item: min(point[1] for point in item[0]))
        if min(point[1] for point in item[0]) < height * name_fraction
        and len(re.findall("[a-zA-Zа-яА-ЯёЁ]", item[1])) >= 3
    ]
    fields = {"product_name": " ".join(names)}
    discounts = re.findall(r"-?\s*\d{1,2}\s*%", text)
    if len(set(discounts)) == 1:
        fields["discount_amount"] = re.sub(r"\s", "", discounts[0])
    date = re.search(r"\b\d{2}\.\d{2}\.\d{4}\s+\d{1,2}:\d{2}\b", text)
    if date:
        fields["print_datetime"] = date.group()
    # Do not assign arbitrary numbers to card/default prices without a label.
    for field, label in [
        ("price_default", r"без\s*карт\w*"),
        ("price_card", r"(?:по|с)\s*карт\w*"),
    ]:
        found = re.search(label + r"\s*[:\-]?\s*(\d{1,6}[.,]\d{2})", text, re.I)
        if found:
            fields[field] = found.group(1).replace(",", ".")
    codes = {value for value in re.findall(r"(?<!\d)\d{8,14}(?!\d)", text) if valid_gtin(value)}
    return fields, codes


def prominent_price(lines, height, width, confidence):
    """Red-card template heuristic; no implicit .00 when cents are missing."""
    numbers = []
    for bbox, text, score in lines:
        token = text.strip().replace(",", ".")
        if score < confidence or not re.fullmatch(r"\d{1,5}(?:\.\d{2})?", token):
            continue
        left = min(point[0] for point in bbox)
        right = max(point[0] for point in bbox)
        top = min(point[1] for point in bbox)
        bottom = max(point[1] for point in bbox)
        numbers.append((bottom - top, token, left, right, top, bottom))
    large = [
        item
        for item in numbers
        if item[0] >= height * 0.12
        and item[4] >= height * 0.40
        and (item[2] + item[3]) / 2 >= width * 0.25
    ]
    if not large:
        return ""
    rubles = max(large)
    if "." in rubles[1]:
        return rubles[1]
    cents = [
        item
        for item in numbers
        if len(item[1]) == 2
        and "." not in item[1]
        and item[0] < rubles[0] * 0.6
        and (item[2] + item[3]) / 2 > (rubles[2] + rubles[3]) / 2
        and rubles[4] - rubles[0] * 0.2 <= (item[4] + item[5]) / 2 <= rubles[4] + rubles[0] * 0.6
    ]
    return rubles[1] + "." + cents[0][1] if len(cents) == 1 else ""


@hydra.main(version_base=None, config_path="../configs", config_name="cpu_ocr")
def main(cfg):
    import cv2
    import easyocr
    import torch
    import zxingcpp

    if cfg.cpu_threads < 1 or cfg.per_video < 0 or cfg.max_side < 1:
        raise ValueError("Invalid resource or sampling configuration")
    torch.set_num_threads(cfg.cpu_threads)
    cv2.setNumThreads(cfg.cpu_threads)
    inputs = resolve(cfg.inputs)
    output = resolve(cfg.output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    OmegaConf.save(cfg, output / "config.yaml")
    rows = [json.loads(line) for line in inputs.read_text(encoding="utf-8").splitlines()]
    connection = catalogue(resolve(cfg.catalogue_csv), resolve(cfg.catalogue_db), cfg.encoding)
    reader = easyocr.Reader(
        ["ru", "en"],
        gpu=False,
        quantize=True,
        verbose=False,
        model_storage_directory=str(resolve(cfg.model_dir)),
        user_network_directory=str(resolve(cfg.model_dir) / "user_network"),
    )
    counts = Counter()
    start = time.perf_counter()
    with (
        (output / "observations.csv").open("w", encoding="utf-8-sig", newline="") as stream,
        (output / "details.jsonl").open("w", encoding="utf-8") as details,
    ):
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            if cfg.per_video and counts[row["filename"]] >= cfg.per_video:
                continue
            tick = time.perf_counter()
            image_path = (inputs.parent / row["image"]).resolve()
            image_path.relative_to(inputs.parent.resolve())
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Unreadable image: {image_path}")
            if cfg.rotation_ccw == 90:
                image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            elif cfg.rotation_ccw == 180:
                image = cv2.rotate(image, cv2.ROTATE_180)
            elif cfg.rotation_ccw == 270:
                image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            elif cfg.rotation_ccw != 0:
                raise ValueError("rotation_ccw must be 0, 90, 180 or 270")
            scale = min(cfg.scale, cfg.max_side / max(image.shape[:2]))
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            decoded = zxingcpp.read_barcodes(image)
            codes, qr_values, conflicts = set(), {}, set()
            for code in decoded:
                if str(code.format) in ("EAN13", "EAN8", "UPCA", "UPCE") and valid_gtin(code.text):
                    codes.add(code.text)
                for field, value in qr_fields(code.text).items():
                    if field in qr_values and qr_values[field] != value:
                        conflicts.add(field)
                    qr_values[field] = value
            for field in conflicts:
                qr_values.pop(field, None)
            if valid_gtin(qr_values.get("qr_code_barcode", "")):
                codes.add(qr_values["qr_code_barcode"])
            lines = reader.readtext(
                image, detail=1, paragraph=False, decoder="greedy", batch_size=1, workers=0
            )
            fields, ocr_codes = parse_text(lines, image.shape[0], cfg.name_fraction, cfg.confidence)
            if cfg.prominent_price_field:
                if cfg.prominent_price_field not in ("price_card", "price_default"):
                    raise ValueError("Unsupported prominent price field")
                if not fields.get(cfg.prominent_price_field):
                    fields[cfg.prominent_price_field] = prominent_price(
                        lines, image.shape[0], image.shape[1], cfg.price_confidence
                    )
            codes.update(ocr_codes)
            matched, suggestions = match_catalog(
                connection, codes, fields["product_name"], cfg.top_k
            )
            record = dict.fromkeys(FIELDS, "")
            record.update(fields)
            record.update(qr_values)
            if len(codes) == 1:
                record["barcode"] = next(iter(codes))
            if matched:
                record["product_name"] = matched["fullname"]
                record["barcode"] = matched["code"]
            record.update(filename=row["filename"], frame_timestamp=row["frame_timestamp"])
            record.update(dict(zip(["x_min", "y_min", "x_max", "y_max"], row["bbox_xyxy"])))
            writer.writerow(record)
            details.write(
                json.dumps(
                    dict(
                        input=row,
                        fields=record,
                        raw_ocr=[
                            dict(
                                text=text,
                                confidence=float(score),
                                bbox=[
                                    [float(coordinate) for coordinate in point] for point in bbox
                                ],
                            )
                            for bbox, text, score in lines
                        ],
                        decoded_codes=[
                            dict(text=code.text, format=str(code.format)) for code in decoded
                        ],
                        exact_catalog_match=matched,
                        catalog_suggestions=suggestions,
                        seconds=time.perf_counter() - tick,
                    ),
                    ensure_ascii=False,
                )
                + "\n"
            )
            counts[row["filename"]] += 1
            counts["__exact_matches"] += int(matched is not None)
            print(f"{row['filename']} {row['image']} {time.perf_counter()-tick:.2f}s", flush=True)
    connection.close()
    write_json(
        output / "summary.json",
        dict(
            counts=dict(counts),
            seconds=time.perf_counter() - start,
            gpu=False,
            input_sha256=checksum(inputs),
            catalogue_sha256=checksum(resolve(cfg.catalogue_csv)),
            note="One row per crop observation; not unique tracked price tags. Empty means unknown.",
        ),
    )


if __name__ == "__main__":
    main()
