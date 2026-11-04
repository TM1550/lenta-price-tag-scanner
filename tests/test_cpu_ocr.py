"""Safety contracts for catalogue lookup and field extraction."""

import tempfile
import unittest
from pathlib import Path

from lenta_price_tag_scanner.cpu_ocr import (
    catalogue,
    match_catalog,
    parse_text,
    prominent_price,
    qr_fields,
    valid_gtin,
)


class ExtractionTests(unittest.TestCase):
    def test_gtin(self):
        self.assertTrue(valid_gtin("4006381333931"))
        self.assertFalse(valid_gtin("4006381333932"))
        self.assertFalse(valid_gtin("abc4006381333931"))

    def test_qr_mapping_is_not_zone_code(self):
        self.assertEqual(
            qr_fields('{"b":"4006381333931","p1":"10.50"}'),
            {"qr_code_barcode": "4006381333931", "price1_qr": "10.50"},
        )
        self.assertEqual(
            qr_fields("https://example.org/?b=4006381333931")["qr_code_barcode"], "4006381333931"
        )
        self.assertEqual(qr_fields("not a QR payload"), {})

    def test_unlabelled_price_is_not_assigned(self):
        fields, codes = parse_text(
            [([[0, 0], [30, 0], [30, 10], [0, 10]], "123.45", 0.99)], 100, 0.4, 0.25
        )
        self.assertNotIn("price_card", fields)
        self.assertFalse(codes)

    def test_cents_required_for_template_price(self):
        rubles = ([[50, 50], [120, 50], [120, 90], [50, 90]], "129", 0.99)
        cents = ([[120, 52], [140, 52], [140, 62], [120, 62]], "99", 0.99)
        self.assertEqual(prominent_price([rubles], 100, 150, 0.7), "")
        self.assertEqual(prominent_price([rubles, cents], 100, 150, 0.7), "129.99")
        self.assertEqual(
            prominent_price([rubles, cents], 100, 150, 0.7, dict(min_top_fraction=0.95)), ""
        )

    def test_ambiguous_catalogue_never_autofills(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "catalog.csv"
            path.write_text(
                "fullname;code\nMilk;4006381333931\nOther;4006381333931\nBread;123\n",
                encoding="cp1251",
            )
            connection = catalogue(path, root / "index.sqlite", "cp1251")
            match, candidates = match_catalog(connection, {"4006381333931"}, "Milk", 5)
            self.assertIsNone(match)
            self.assertTrue(candidates)
            match, _ = match_catalog(connection, {"123"}, "", 5)
            self.assertEqual(match["fullname"], "Bread")
            connection.close()
            path.write_text("fullname;code\nChanged;123\n", encoding="cp1251")
            with self.assertRaises(ValueError):
                catalogue(path, root / "index.sqlite", "cp1251")


if __name__ == "__main__":
    unittest.main()
