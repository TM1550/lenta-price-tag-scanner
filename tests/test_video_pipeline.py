"""Coverage must not reward metadata or dropped tracks; fusion must preserve ambiguity."""

import unittest

from omegaconf import OmegaConf
from PIL import Image

from lenta_price_tag_scanner.crop_quality import preprocess, quality, rejection_reasons
from lenta_price_tag_scanner.field_extraction import (
    CONTENT_FIELDS,
    compatible_observations,
    coverage,
    fuse,
    text_fields,
)


class PipelineTests(unittest.TestCase):
    def test_identity_switch_cannot_mix_products(self):
        observations = [
            dict(image="a", fields={"product_name": "Молоко марка А", "barcode": "4006381333931"}),
            dict(image="b", fields={"product_name": "Вино марка Б", "price_card": "99.99"}),
            dict(image="c", fields={"barcode": "4006381333931", "price_card": "12.00"}),
        ]
        selected, excluded = compatible_observations(observations)
        self.assertEqual([row["image"] for row in selected], ["a", "c"])
        self.assertEqual(excluded, ["b"])

    def test_coverage_excludes_metadata_and_unknown(self):
        result = coverage(
            [
                dict(
                    filename="a",
                    frame_timestamp=0,
                    x_min=5,
                    barcode="нет",
                    product_name="Milk",
                    price_card="12.00",
                ),
                dict(product_name="", price_card=None),
            ]
        )
        self.assertEqual(len(CONTENT_FIELDS), 23)
        self.assertEqual(result["possible_cells"], 46)
        self.assertEqual(result["filled_cells"], 2)
        self.assertAlmostEqual(result["coverage"], 2 / 46)
        self.assertEqual(coverage([])["coverage"], 0)

    def test_conflicting_prices_remain_empty(self):
        observations = [
            dict(image="a", fields={"price_card": "10.00"}, provenance={}),
            dict(image="b", fields={"price_card": "20.00"}, provenance={}),
        ]
        fields, _, conflicts = fuse(observations)
        self.assertNotIn("price_card", fields)
        self.assertIn("price_card", conflicts)

    def test_decoded_barcode_wins_over_ocr(self):
        fields, _, _ = fuse(
            [
                dict(
                    image="a",
                    fields={"barcode": "4006381333931"},
                    provenance={"barcode": "decoded_barcode"},
                ),
                dict(image="b", fields={"barcode": "12345678"}, provenance={"barcode": "ocr_text"}),
            ]
        )
        self.assertEqual(fields["barcode"], "4006381333931")

    def test_no_invented_price_and_invalid_date(self):
        self.assertNotIn("price_card", text_fields("129 99"))
        self.assertNotIn("print_datetime", text_fields("31.02.2026 10:20"))
        self.assertEqual(text_fields("по карте 129,99")["price_card"], "129.99")
        self.assertNotIn("barcode", text_fields("4006381333932"))

    def test_blank_crop_rejected(self):
        image = Image.new("RGB", (100, 150), "white")
        cfg = OmegaConf.create(
            dict(min_side=40, min_sharpness=15, min_contrast=12, max_flat_fraction=0.98)
        )
        self.assertIn("exposure", rejection_reasons(quality(image), cfg))
        self.assertEqual(preprocess(image, "rectify").size, image.size)


if __name__ == "__main__":
    unittest.main()
