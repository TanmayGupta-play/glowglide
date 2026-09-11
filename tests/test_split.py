"""Synthetic tests for the global temporal boundary."""

import unittest

import pandas as pd
from pandas.testing import assert_frame_equal

from glowguide.split import temporal_split


class TemporalSplitTests(unittest.TestCase):
    def test_global_boundary_and_no_input_mutation(self):
        frame = pd.DataFrame({
            "author_id": ["u1", "u2", "u1", "u2", "new"],
            "submission_time": ["2024-01-05", "2024-01-01", "2024-01-03", "2024-01-02", "2024-01-04"],
        })
        before = frame.copy(deep=True)
        result = temporal_split(frame)
        self.assertEqual(result.cutoff, pd.Timestamp("2024-01-04 04:48:00"))
        self.assertTrue(result.train["submission_time"].le(result.cutoff).all())
        self.assertTrue(result.test["submission_time"].gt(result.cutoff).all())
        self.assertLess(result.train["submission_time"].max(), result.test["submission_time"].min())
        self.assertFalse(set(result.train.index) & set(result.test.index))
        self.assertEqual(len(result.train) + len(result.test), len(frame))
        self.assertIn("new", result.train["author_id"].tolist())
        assert_frame_equal(frame, before)
        result.train.loc[result.train.index[0], "author_id"] = "changed"
        assert_frame_equal(frame, before)

    def test_cutoff_ties_stay_in_train(self):
        frame = pd.DataFrame({"submission_time": pd.to_datetime([
            "2024-01-01", "2024-01-02", "2024-01-02", "2024-01-03",
        ])})
        result = temporal_split(frame, 0.5)
        self.assertEqual(result.cutoff, pd.Timestamp("2024-01-02"))
        self.assertEqual(len(result.train), 3)
        self.assertEqual(len(result.test), 1)

    def test_invalid_quantile(self):
        frame = pd.DataFrame({"submission_time": ["2024-01-01"]})
        for quantile in (0, 1, -0.1, 1.1, float("nan"), None, "0.8", True):
            with self.subTest(quantile=quantile), self.assertRaisesRegex(ValueError, "split_quantile"):
                temporal_split(frame, quantile)

    def test_empty_and_invalid_dates(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            temporal_split(pd.DataFrame())
        with self.assertRaisesRegex(ValueError, "submission_time"):
            temporal_split(pd.DataFrame({"other": [1]}))
        for value in (None, "not a date", pd.NaT):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "invalid or missing"):
                temporal_split(pd.DataFrame({"submission_time": [value]}))

    def test_identical_timestamps_do_not_force_an_artificial_test_split(self):
        result = temporal_split(pd.DataFrame({"submission_time": ["2024-01-01"] * 3}))
        self.assertEqual(len(result.train), 3)
        self.assertTrue(result.test.empty)
