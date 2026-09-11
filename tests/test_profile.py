"""Synthetic checks for TRAIN profile inference and hierarchical affinity."""

import unittest

import pandas as pd
from pandas.testing import assert_frame_equal

from glowguide.profile import infer_user_profiles
from glowguide.recommenders.profile import SkinProfileRecommender
from glowguide.split import temporal_split


def frame(rows):
    return pd.DataFrame(rows, columns=["author_id", "product_id", "positive", "skin_type", "skin_tone"]).assign(submission_time="2024-01-01")


class ProfileTests(unittest.TestCase):
    def setUp(self):
        self.train = frame([
            ("d1", "A", 1, "dry", "fair"), ("d2", "A", 1, "dry", "fair"),
            ("o1", "A", 0, "oily", "deep"), ("o2", "A", 0, "oily", "deep"),
            ("d1", "B", 0, "dry", "fair"), ("d2", "B", 0, "dry", "fair"),
            ("o1", "B", 1, "oily", "deep"), ("o2", "B", 1, "oily", "deep"),
        ])

    def test_independent_modes_and_latest_tied_value(self):
        train = frame([
            ("u", "A", 1, "dry", "fair"), ("u", "B", 0, "dry", "deep"),
            ("u", "C", 1, "oily", "deep"), ("u", "D", 1, "oily", "fair"),
            ("u", "E", 1, "normal", "deep"),
        ])
        train["submission_time"] = pd.date_range("2024-01-01", periods=5)
        profile = infer_user_profiles(train).loc["u"]
        self.assertEqual(profile["skin_type"], "oily")
        self.assertEqual(profile["skin_tone"], "deep")
        train.loc[4, "skin_type"] = "dry"
        train.loc[4, "submission_time"] = pd.Timestamp("2023-01-01")
        self.assertEqual(infer_user_profiles(train).loc["u", "skin_type"], "dry")
        self.assertEqual(infer_user_profiles(train.iloc[:4]).loc["u", "skin_tone"], "fair")

    def test_train_only_inference_and_statistics(self):
        future = frame([("d1", "FUTURE", 0, "oily", "deep")]).assign(submission_time="2025-01-01")
        split = temporal_split(pd.concat([self.train, future], ignore_index=True), 0.8)
        model = SkinProfileRecommender().fit(split.train)
        expected = SkinProfileRecommender().fit(self.train)
        self.assertEqual(model.explain_score("d1", "A"), expected.explain_score("d1", "A"))
        self.assertEqual(model.explain_score("d1", "A")["skin_type"], "dry")
        self.assertNotIn("FUTURE", model.candidate_product_ids)

    def test_missing_attributes_and_overall_fallback(self):
        extra = frame([
            ("type", "A", 1, "dry", None), ("tone", "A", 1, None, "fair"),
            ("none", "A", 0, None, None),
        ])
        model = SkinProfileRecommender().fit(pd.concat([self.train, extra], ignore_index=True))
        for user, available in [("type", {"overall", "skin_type"}), ("tone", {"overall", "skin_tone"}), ("none", {"overall"})]:
            explanation = model.explain_score(user, "B")
            signals = {key: value for key, value in explanation["signals"].items() if value is not None}
            self.assertEqual(set(signals), available)
            self.assertAlmostEqual(sum(v["weight"] for v in signals.values()), 1)
            self.assertEqual(model.recommend({"A"}, 10, user_id=user), ["B"])
        missing = model.explain_score("none", "B")
        self.assertIsNone(missing["skin_type"])
        self.assertIsNone(missing["skin_tone"])
        self.assertEqual(missing["final_score"], missing["signals"]["overall"]["smoothed_rate"])
        self.assertEqual(model.recommend(set(), user_id="unknown"), model.recommend(set(), user_id="none"))
        absent = infer_user_profiles(extra.drop(columns=["skin_type", "skin_tone"]))
        self.assertTrue(absent.isna().all().all())

    def test_exact_affinities_and_personalized_ranking(self):
        model = SkinProfileRecommender().fit(self.train)
        dry = model.explain_score("d1", "A")["signals"]["exact"]
        oily = model.explain_score("o1", "A")["signals"]["exact"]
        self.assertGreater(dry["smoothed_rate"], oily["smoothed_rate"])
        self.assertEqual(dry["positive_count"], 2)
        self.assertEqual(oily["interaction_count"], 2)
        self.assertEqual(model.recommend(set(), 2, user_id="d1"), ["A", "B"])
        self.assertEqual(model.recommend(set(), 2, user_id="o1"), ["B", "A"])

    def test_one_review_smoothing_and_exact_formula(self):
        train = frame([("u", "A", 1, "dry", "fair"), ("v", "B", 0, "oily", "deep")])
        explanation = SkinProfileRecommender().fit(train).explain_score("u", "A")
        overall = (1 + 20 * 0.5) / 21
        single = (1 + 20 * overall) / 21
        exact = (1 + 40 * single) / 41
        self.assertAlmostEqual(explanation["signals"]["overall"]["smoothed_rate"], overall)
        self.assertAlmostEqual(explanation["signals"]["skin_type"]["smoothed_rate"], single)
        self.assertAlmostEqual(explanation["signals"]["exact"]["smoothed_rate"], exact)
        self.assertLess(exact, 0.6)
        self.assertAlmostEqual(explanation["final_score"], 0.1 * overall + 0.4 * single + 0.5 * exact)

    def test_negative_interactions_lower_affinity(self):
        before = SkinProfileRecommender().fit(self.train).explain_score("d1", "A")["final_score"]
        extra = frame([("d3", "A", 0, "dry", "fair")])
        after = SkinProfileRecommender().fit(pd.concat([self.train, extra])).explain_score("d1", "A")["final_score"]
        self.assertLess(after, before)

    def test_seen_ties_no_duplicates_and_determinism(self):
        model = SkinProfileRecommender().fit(self.train)
        self.assertEqual(model.recommend({"A", "B"}, 10, user_id="d1"), [])
        self.assertEqual(model.recommend({"B"}, 10, user_id="d1"), ["A"])
        self.assertEqual(model.recommend(set(), 10, user_id="unknown"), ["A", "B"])
        reverse = SkinProfileRecommender().fit(self.train.iloc[::-1])
        self.assertEqual(model.recommend(set(), user_id="d1"), reverse.recommend(set(), user_id="d1"))
        self.assertEqual(model.explain_score("d1", "A"), reverse.explain_score("d1", "A"))
        result = model.recommend(set(), 10, user_id="d1")
        self.assertEqual(len(result), len(set(result)))

    def test_explanation_matches_rank_and_backoff(self):
        model = SkinProfileRecommender().fit(self.train)
        ranking = model.recommend(set(), user_id="d1")
        scores = []
        for product in ranking:
            explanation = model.explain_score("d1", product)
            score = sum(value["weight"] * value["smoothed_rate"] for value in explanation["signals"].values() if value)
            self.assertAlmostEqual(score, explanation["final_score"])
            scores.append(score)
        self.assertEqual(scores, sorted(scores, reverse=True))
        extra = frame([("x", "C", 0, None, None)])
        model.fit(pd.concat([self.train, extra]))
        signals = model.explain_score("d1", "C")["signals"]
        self.assertEqual(signals["exact"]["interaction_count"], 0)
        self.assertEqual(signals["exact"]["smoothed_rate"], signals["overall"]["smoothed_rate"])

    def test_no_mutation_and_clear_errors(self):
        before = self.train.copy(deep=True)
        model = SkinProfileRecommender()
        with self.assertRaisesRegex(RuntimeError, "Fit"):
            model.recommend(set())
        model.fit(self.train)
        assert_frame_equal(before, self.train)
        with self.assertRaisesRegex(ValueError, "train catalog"):
            model.explain_score("d1", "FUTURE")
        with self.assertRaisesRegex(ValueError, "k"):
            model.recommend(set(), 0)
        with self.assertRaisesRegex(ValueError, "Nonempty"):
            model.fit(self.train.iloc[:0])
        with self.assertRaisesRegex(ValueError, "timestamps"):
            infer_user_profiles(self.train.assign(submission_time="bad"))
