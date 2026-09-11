"""Train-only popularity ranking and seen-product exclusion."""

import unittest

import pandas as pd
from pandas.testing import assert_frame_equal

from glowguide.recommenders import MostPopularRecommender


class PopularityTests(unittest.TestCase):
    def test_unique_positive_user_counts_and_seen_products(self):
        train = pd.DataFrame([
            ("u1", "A", 1), ("u2", "A", 1), ("u3", "A", 1),
            ("u1", "B", 1), ("u2", "B", 1),
            ("u1", "C", 1), ("u1", "C", 1), ("u1", "C", 1),
            ("u4", "C", 0), ("u5", "C", 0), ("u6", "C", 0),
            ("u7", "D", 0),
        ], columns=["author_id", "product_id", "positive"])
        before = train.copy(deep=True)
        model = MostPopularRecommender().fit(train)
        self.assertEqual(model.recommend(set(), 10), ["A", "B", "C", "D"])
        self.assertEqual(model.recommend({"A", "C"}, 10), ["B", "D"])
        self.assertEqual(model.recommend(set(), 2), ["A", "B"])
        self.assertEqual(model.recommend({"A", "B", "C", "D"}, 10), [])
        self.assertEqual(model.candidate_product_ids, frozenset({"A", "B", "C", "D"}))
        assert_frame_equal(train, before)

    def test_deterministic_ties_and_zero_positive_products(self):
        train = pd.DataFrame([
            ("u1", "B", 1), ("u2", "A", 1), ("u3", "D", 0), ("u4", "C", 0),
        ], columns=["author_id", "product_id", "positive"])
        for frame in (train, train.iloc[::-1]):
            result = MostPopularRecommender().fit(frame).recommend(set(), 10)
            self.assertEqual(result, ["A", "B", "C", "D"])
            self.assertEqual(len(result), len(set(result)))
        train["positive"] = 0
        self.assertEqual(MostPopularRecommender().fit(train).recommend(set(), 10), ["A", "B", "C", "D"])

    def test_fit_and_recommend_errors(self):
        model = MostPopularRecommender()
        with self.assertRaisesRegex(RuntimeError, "Fit"):
            model.recommend(set(), 10)
        with self.assertRaisesRegex(ValueError, "k"):
            model.recommend(set(), 0)
        with self.assertRaisesRegex(ValueError, "empty"):
            model.fit(pd.DataFrame(columns=["author_id", "product_id", "positive"]))
        with self.assertRaisesRegex(ValueError, "required columns"):
            model.fit(pd.DataFrame({"product_id": ["A"]}))

    def test_refitting_replaces_the_previous_catalog(self):
        model = MostPopularRecommender()
        for product in ("A", "B"):
            model.fit(pd.DataFrame({"author_id": ["u1"], "product_id": [product], "positive": [1]}))
            self.assertEqual(model.recommend(set(), 10), [product])
