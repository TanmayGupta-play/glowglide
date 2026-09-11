"""Manually verifiable binary ranking metrics."""

from math import log2
import unittest

from glowguide.metrics import (
    average_precision_at_k,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


METRICS = (precision_at_k, recall_at_k, hit_rate_at_k, ndcg_at_k, average_precision_at_k)


class RankingMetricsTests(unittest.TestCase):
    def test_manual_example(self):
        recommended, relevant = ["A", "B", "C", "D"], {"B", "D"}
        self.assertEqual(precision_at_k(recommended, relevant, 4), 0.5)
        self.assertEqual(recall_at_k(recommended, relevant, 4), 1.0)
        self.assertEqual(hit_rate_at_k(recommended, relevant, 4), 1.0)
        self.assertAlmostEqual(ndcg_at_k(recommended, relevant, 4), (1 / log2(3) + 1 / log2(5)) / (1 + 1 / log2(3)))
        self.assertEqual(average_precision_at_k(recommended, relevant, 4), 0.5)
        self.assertEqual(precision_at_k(recommended, relevant, 2), 0.5)
        self.assertEqual(recall_at_k(recommended, relevant, 2), 0.5)
        self.assertEqual(average_precision_at_k(recommended, relevant, 2), 0.25)
        self.assertEqual(hit_rate_at_k(recommended, relevant, 1), 0.0)

    def test_empty_inputs_return_zero(self):
        for metric in METRICS:
            for recommended, relevant in [([], {"A"}), (["A"], set()), ([], set())]:
                with self.subTest(metric=metric.__name__, recommended=recommended, relevant=relevant):
                    self.assertEqual(metric(recommended, relevant, 10), 0.0)

    def test_invalid_k_raises_even_with_empty_inputs(self):
        for metric in METRICS:
            for k in (0, -1, 1.5, True):
                with self.subTest(metric=metric.__name__, k=k), self.assertRaisesRegex(ValueError, "k"):
                    metric([], set(), k)

    def test_short_lists_and_ap_denominator(self):
        self.assertEqual(precision_at_k(["A"], {"A"}, 10), 0.1)
        self.assertEqual(recall_at_k(["A"], {"A", "B", "C"}, 2), 1 / 3)
        self.assertEqual(average_precision_at_k(["A", "B"], {"A", "B", "C"}, 2), 1.0)
        self.assertEqual(ndcg_at_k(["A", "B"], {"A", "B", "C"}, 2), 1.0)

    def test_duplicates_cannot_inflate_relevance(self):
        for metric in METRICS:
            with self.subTest(metric=metric.__name__):
                score = metric(["A", "A", "B"], {"A", "B"}, 3)
                self.assertGreaterEqual(score, 0)
                self.assertLessEqual(score, 1)
        self.assertEqual(precision_at_k(["A", "A"], {"A"}, 2), 0.5)
        self.assertEqual(average_precision_at_k(["A", "A"], {"A"}, 2), 1.0)
