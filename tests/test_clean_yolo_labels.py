"""Geometry and suppression checks for manually reviewed detection labels."""

import unittest

from lenta_price_tag_scanner.clean_yolo_labels import clean_boxes, iou, parse_label, yolo_line


class CleanLabelTests(unittest.TestCase):
    def box(self, rect, line=1, cls=0):
        return dict(xyxy=rect, line=line, cls=cls)

    def test_mixed_export_converts_polygons_and_keeps_xywh_semantics(self):
        polygon = parse_label("0 0.2 0.1 0.4 0.1 0.3 0.5", 1, 1)
        box = parse_label("0 0.3 0.3 0.2 0.4", 2, 1)
        for expected, actual in zip(polygon["xyxy"], box["xyxy"]):
            self.assertAlmostEqual(expected, actual)
        self.assertEqual(len(yolo_line(polygon).split()), 5)

    def test_nms_keeps_larger_duplicate_and_separate_neighbour(self):
        large = self.box([0.1, 0.1, 0.5, 0.5], line=2)
        duplicate = self.box([0.12, 0.12, 0.48, 0.48])
        neighbour = self.box([0.45, 0.1, 0.8, 0.5], line=3)
        kept, removed = clean_boxes([duplicate, large, neighbour], 1000, 1000, 16, 0.5)
        self.assertEqual([box["line"] for box in kept], [2, 3])
        self.assertEqual(removed[0]["kept_line"], 2)
        self.assertGreater(removed[0]["iou"], 0.5)

    def test_distinct_classes_are_not_suppressed(self):
        boxes = [self.box([0.1, 0.1, 0.5, 0.5], 1, 0), self.box([0.1, 0.1, 0.5, 0.5], 2, 1)]
        self.assertEqual(len(clean_boxes(boxes, 1000, 1000, 16, 0.5)[0]), 2)

    def test_small_box_uses_short_side_in_original_pixels(self):
        thin = self.box([0.1, 0.1, 0.6, 0.105])
        boundary = self.box([0.1, 0.1, 0.3, 0.108], line=2)
        kept, removed = clean_boxes([thin, boundary], 4000, 2000, 16, 0.5)
        self.assertEqual([box["line"] for box in kept], [2])
        self.assertEqual(removed[0]["reason"], "small")

    def test_disjoint_boxes_and_invalid_coordinates(self):
        self.assertEqual(iou([0, 0, 0.1, 0.1], [0.2, 0.2, 0.3, 0.3]), 0)
        for text in ("0 nan 0.1 0.2 0.2", "0 1.1 0.1 0.2 0.2", "2 0.1 0.1 0.2 0.2"):
            with self.assertRaises(ValueError):
                parse_label(text, 1, 1)

    def test_small_filter_precedes_nms(self):
        boxes = [self.box([0.1, 0.1, 0.2, 0.101]), self.box([0.1, 0.1, 0.2, 0.101], 2)]
        kept, removed = clean_boxes(boxes, 4000, 2000, 16, 0.5)
        self.assertEqual(kept, [])
        self.assertEqual([row["reason"] for row in removed], ["small", "small"])


if __name__ == "__main__":
    unittest.main()
