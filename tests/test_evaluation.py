"""Synthetic integration checks for eligibility, relevance, and averaging."""

import unittest
from unittest.mock import patch

import pandas as pd

from glowguide.evaluation import eligible_evaluation_users, evaluate_ranking
from glowguide.recommenders import MostPopularRecommender
from glowguide.split import temporal_split


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        columns = ["author_id", "product_id", "positive"]
        self.train = pd.DataFrame([
            ("u1", "A", 1), ("u1", "B", 1), ("u1", "C", 0),
            ("u2", "A", 1), ("u2", "B", 1), ("u3", "A", 1),
            ("u4", "D", 1), ("u4", "E", 1),
        ], columns=columns)
        self.test = pd.DataFrame([
            ("u1", "D", 1), ("u1", "E", 0), ("u1", "NEW", 1),
            ("u2", "NEW", 1), ("u3", "D", 1),
        ], columns=columns)

    def test_eligibility_relevance_macro_metrics_and_latency(self):
        self.assertEqual(eligible_evaluation_users(self.train, self.test).tolist(), ["u1", "u2"])
        model = MostPopularRecommender().fit(self.train)
        with patch.object(model, "recommend", wraps=model.recommend) as recommend:
            with patch("glowguide.evaluation.perf_counter", side_effect=[0, 0.001]):
                result = evaluate_ranking(model, self.train, self.test, k=2)
        self.assertEqual(recommend.call_count, 1)
        self.assertEqual(recommend.call_args_list[0].args[0], {"A", "B", "C"})
        self.assertEqual(result["history_eligible_users"], 2)
        self.assertEqual(result["servable_evaluation_users"], 1)
        self.assertEqual(result["cold_start_only_users"], 1)
        self.assertEqual(result["cold_start_only_user_rate"], 0.5)
        self.assertEqual(result["eligible_positive_test_interactions"], 3)
        self.assertEqual(result["relevant_test_interactions_used"], 1)
        self.assertEqual(result["positive_test_interactions_outside_train_catalog"], 2)
        self.assertEqual(result["positive_test_interactions_outside_train_catalog_rate"], 2 / 3)
        self.assertEqual(result["precision_at_k"], 0.5)
        for metric in ("recall_at_k", "hit_rate_at_k", "ndcg_at_k", "map_at_k"):
            self.assertEqual(result[metric], 1.0)
        self.assertEqual(result["catalog_coverage_at_k"], 2 / 5)
        self.assertAlmostEqual(result["mean_recommendation_latency_ms"], 1.0)
        self.assertAlmostEqual(result["p95_recommendation_latency_ms"], 1.0)

    def test_coverage_excludes_cold_start_only_user_recommendations(self):
        # Only u1 has seen D, so u1 gets E while cold-start-only u2 would get D.
        train = pd.concat([self.train, pd.DataFrame({
            "author_id": ["u1"], "product_id": ["D"], "positive": [0],
        })], ignore_index=True)
        test = self.test.copy()
        test.loc[test["product_id"].eq("D") & test["author_id"].eq("u1"), "positive"] = 0
        test.loc[test["product_id"].eq("E"), "positive"] = 1
        model = MostPopularRecommender().fit(train)
        self.assertEqual(model.recommend({"A", "B"}, 1), ["D"])
        with patch.object(model, "recommend", wraps=model.recommend) as recommend:
            result = evaluate_ranking(model, train, test, k=1)
        recommend.assert_called_once_with({"A", "B", "C", "D"}, 1)
        self.assertEqual(result["catalog_coverage_at_k"], 1 / 5)

    def test_zero_servable_users_raises_before_recommending(self):
        test = self.test.loc[self.test["product_id"].eq("NEW")]
        self.assertEqual(eligible_evaluation_users(self.train, test).tolist(), ["u1", "u2"])
        model = MostPopularRecommender().fit(self.train)
        with patch.object(model, "recommend", wraps=model.recommend) as recommend:
            with self.assertRaisesRegex(ValueError, "No servable evaluation users"):
                evaluate_ranking(model, self.train, test)
        recommend.assert_not_called()

    def test_negative_history_is_seen_even_if_product_is_popular(self):
        train = pd.DataFrame([
            ("u1", "A", 0), ("u1", "B", 1), ("u1", "C", 1),
            ("u2", "A", 1), ("u3", "A", 1), ("u4", "D", 1),
        ], columns=["author_id", "product_id", "positive"])
        test = pd.DataFrame({"author_id": ["u1"], "product_id": ["D"], "positive": [1]})
        model = MostPopularRecommender().fit(train)
        self.assertEqual(model.recommend(set(), 1), ["A"])
        result = evaluate_ranking(model, train, test, k=1)
        self.assertEqual(result["precision_at_k"], 1.0)

    def test_no_eligible_users_and_invalid_threshold(self):
        model = MostPopularRecommender().fit(self.train)
        with self.assertRaisesRegex(ValueError, "No eligible"):
            evaluate_ranking(model, self.train, self.test, min_train_positives=3)
        for threshold in (0, -1, 1.5, True):
            with self.subTest(threshold=threshold), self.assertRaisesRegex(ValueError, "min_train_positives"):
                eligible_evaluation_users(self.train, self.test, threshold)

    def test_bad_candidate_catalog_and_recommendations_are_rejected(self):
        model = MostPopularRecommender().fit(pd.concat([self.train, self.test]))
        with self.assertRaisesRegex(ValueError, "candidate catalog"):
            evaluate_ranking(model, self.train, self.test)
        model.fit(self.train)
        for returned in (["D", "D"], ["A"], ["NEW"]):
            with self.subTest(returned=returned), patch.object(model, "recommend", return_value=returned):
                with self.assertRaisesRegex(ValueError, "Recommender returned"):
                    evaluate_ranking(model, self.train, self.test)

    def test_temporal_pipeline_does_not_learn_future_popularity(self):
        frame = pd.concat([
            self.train.assign(submission_time="2024-01-01"),
            self.test.assign(submission_time="2024-02-01"),
        ], ignore_index=True)
        split = temporal_split(frame, 0.5)
        model = MostPopularRecommender().fit(split.train)
        ranking = model.recommend(set(), 10)
        self.assertEqual(ranking, ["A", "B", "D", "E", "C"])
        self.assertNotIn("NEW", model.candidate_product_ids)
        evaluate_ranking(model, split.train, split.test, k=2)
        self.assertEqual(model.recommend(set(), 10), ranking)
