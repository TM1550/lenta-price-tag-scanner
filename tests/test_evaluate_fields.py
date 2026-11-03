"""Field scoring must not reward missing detections or reuse predictions."""

import unittest

from lenta_price_tag_scanner.evaluate_fields import score


class FieldScoreTests(unittest.TestCase):
    def setUp(self):
        self.row = dict(
            frame_timestamp="100",
            x_min="0",
            y_min="0",
            x_max="10",
            y_max="10",
            price="12,50",
            product_name="Товар",
        )

    def test_decimal_normalization(self):
        self.assertEqual(score([self.row], [dict(self.row, price="12.50")])["correct"], 2)

    def test_missing_detection_stays_in_denominator(self):
        result = score([self.row], [])
        self.assertEqual((result["correct"], result["expected"]), (0, 2))

    def test_prediction_cannot_match_two_labels(self):
        result = score([self.row, self.row], [self.row])
        self.assertEqual((result["correct"], result["expected"]), (2, 4))

    def test_other_frame_does_not_match(self):
        self.assertEqual(score([self.row], [dict(self.row, frame_timestamp="200")])["correct"], 0)


if __name__ == "__main__":
    unittest.main()
