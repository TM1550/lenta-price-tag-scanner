"""Track association must not use OCR answers or silently reuse a CSV prediction."""

import unittest

from lenta_price_tag_scanner.evaluate_tracks import associate, compare_fields, link_rows


class TrackEvaluationTests(unittest.TestCase):
    def observation(self, track=1, bbox=None, pts=0):
        return dict(track_id=track, bbox_xyxy=bbox or [0, 0, 100, 100], pts_ms=pts)

    def target(self, identity="v_0000", price="10.00"):
        return dict(
            id=identity,
            video="v",
            bbox=[0, 0, 100, 100],
            timestamp=0,
            fields=dict(
                price_card=price,
                filename="v",
                frame_timestamp="0",
                x_min="0",
                y_min="0",
                x_max="100",
                y_max="100",
            ),
        )

    def prediction(self, index=0, price="10.00", rank=1):
        return dict(
            index=index,
            video="v",
            bbox=[0, 0, 100, 100],
            pts_ms=1000,
            frame_index=20,
            quality_rank=rank,
            pipeline_track_id=99,
            fields=dict(price_card=price),
        )

    def history(self):
        return {"v": {0: [self.observation()], 20: [self.observation(pts=1000)]}}

    def test_same_track_links_different_times_and_cross_run_ids(self):
        links, _ = link_rows(
            [self.target()], [self.prediction()], self.history(), {"v": [0, 1000]}, 0.5, 0.1, 100
        )
        self.assertEqual(links[0]["prediction"]["pipeline_track_id"], 99)
        self.assertEqual(links[0]["link"]["track_id"], 1)
        self.assertEqual(compare_fields(links)[2]["correct_fill"], 1)

    def test_nearest_frame_is_used_not_best_iou_from_window(self):
        history = {
            0: [self.observation(bbox=[200, 200, 300, 300], pts=0)],
            1: [self.observation(pts=50)],
        }
        result = associate([0, 0, 100, 100], 1, history, 1000)
        self.assertEqual(result["status"], "no_overlapping_track")

    def test_ambiguous_candidates_are_not_credited(self):
        history = {0: [self.observation(1), self.observation(2)]}
        self.assertEqual(
            associate([0, 0, 100, 100], 0, history, 1000)["status"], "ambiguous_tracks"
        )

    def test_out_of_range_gt_is_not_clamped_to_last_frame(self):
        self.assertEqual(
            associate([0, 0, 100, 100], 1001, self.history()["v"], 1000)["status"],
            "timestamp_outside_video",
        )

    def test_duplicate_csv_rows_are_selected_by_quality_not_correct_answer(self):
        links, _ = link_rows(
            [self.target()],
            [self.prediction(0, "10", 1), self.prediction(1, "20", 2)],
            self.history(),
            {"v": [0, 1000]},
            0.5,
            0.1,
            100,
        )
        self.assertEqual(links[0]["prediction"]["index"], 1)
        self.assertEqual(links[0]["duplicate_prediction_indices"], [0])
        self.assertEqual(compare_fields(links)[2]["correct"], 0)

    def test_multiple_gt_anchors_on_one_track_require_review(self):
        links, _ = link_rows(
            [self.target(), self.target("v_0001")],
            [self.prediction()],
            self.history(),
            {"v": [0, 1000]},
            0.5,
            0.1,
            100,
        )
        report = compare_fields(links)[2]
        self.assertEqual(report["expected"], 2)
        self.assertEqual(report["correct"], 0)
        self.assertTrue(all("prediction" not in row for row in links))

    def test_missing_prediction_stays_in_denominator(self):
        links, _ = link_rows([self.target()], [], self.history(), {"v": [0, 1000]}, 0.5, 0.1, 100)
        self.assertEqual(compare_fields(links)[2]["expected"], 1)
        self.assertEqual(compare_fields(links)[2]["correct"], 0)

    def test_video_identity_partitions_tracks(self):
        second = self.prediction()
        second["video"] = "other"
        links, _ = link_rows(
            [self.target()],
            [second],
            dict(self.history(), other=self.history()["v"]),
            {"v": [0, 1000], "other": [0, 1000]},
            0.5,
            0.1,
            100,
        )
        self.assertNotIn("prediction", links[0])

    def test_color_does_not_hide_price_errors_in_diagnostic_groups(self):
        target = self.target()
        target["fields"]["color"] = "red"
        prediction = self.prediction(price="20.00")
        prediction["fields"]["color"] = "red"
        links, _ = link_rows(
            [target], [prediction], self.history(), {"v": [0, 1000]}, 0.5, 0.1, 100
        )
        report = compare_fields(links)[2]
        self.assertEqual(report["correct_fill"], 0.5)
        self.assertEqual(report["field_groups"]["without_color"]["correct_fill"], 0)
        self.assertEqual(report["field_groups"]["printed_prices"]["correct_fill"], 0)


if __name__ == "__main__":
    unittest.main()
