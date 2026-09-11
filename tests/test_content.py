"""Synthetic content ranking, profile, explanation, and isolation tests."""

import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from glowguide.evaluation import evaluate_ranking
from glowguide.features import build_product_features
from glowguide.recommenders.content import TfidfContentRecommender
from glowguide.recommenders import MostPopularRecommender


class ContentTests(unittest.TestCase):
    def setUp(self):
        self.products = pd.DataFrame({
            "product_id": ["H", "A", "B", "N", "FUTURE"],
            "product_name": ["Hydrating hyaluronic acid serum", "Hyaluronic hydrating moisturizer",
                             "Charcoal clay oil control mask", "Charcoal clay cleanser", "futureonlytoken"],
            "ingredients": [None] * 5, "highlights": [None] * 5,
        })
        self.train = pd.DataFrame([
            ("u1", "H", 5, 1), ("u1", "N", 1, 0),
            ("other", "A", 4, 1), ("other", "B", 4, 1),
        ], columns=["author_id", "product_id", "rating", "positive"])

    def test_related_product_seen_exclusion_and_train_catalog(self):
        model = TfidfContentRecommender().fit(self.train, self.products)
        result = model.recommend({"H", "N"}, 10, user_id="u1")
        self.assertEqual(result, ["A", "B"])
        self.assertEqual(len(result), len(set(result)))
        self.assertEqual(model.candidate_product_ids, frozenset({"H", "A", "B", "N"}))
        self.assertNotIn("futureonlytoken", model.vectorizer.vocabulary_)
        self.assertEqual(model.recommend(model.candidate_product_ids, 10, user_id="u1"), [])

    def test_weight_five_is_twice_weight_four(self):
        products = pd.DataFrame({"product_id": ["A", "B"], "product_name": ["niacinamide", "hyaluronic"]})
        train = pd.DataFrame({"author_id": ["u", "u"], "product_id": ["A", "B"], "rating": [5, 4], "positive": [1, 1]})
        model = TfidfContentRecommender().fit(train, products)
        vector = model.user_profile("u").toarray().ravel()
        vocabulary = model.vectorizer.vocabulary_
        self.assertAlmostEqual(vector[vocabulary["niacinamide"]], 2 * vector[vocabulary["hyaluronic"]])
        self.assertAlmostEqual(np.linalg.norm(vector), 1.0)

    def test_negative_history_does_not_affect_profile(self):
        model = TfidfContentRecommender().fit(self.train, self.products)
        changed = self.train.copy()
        changed.loc[changed["product_id"].eq("N"), "author_id"] = "another_user"
        other = TfidfContentRecommender().fit(changed, self.products)
        np.testing.assert_allclose(model.user_profile("u1").toarray(), other.user_profile("u1").toarray())
        self.assertEqual(model.user_profile("u1")[0, model.vectorizer.vocabulary_["charcoal"]], 0)

    def test_ties_and_refits_are_deterministic(self):
        products = self.products.copy()
        products.loc[products["product_id"].isin(["A", "B"]), "product_name"] = "hydrating hyaluronic"
        first = TfidfContentRecommender().fit(self.train, products)
        second = TfidfContentRecommender().fit(self.train.iloc[::-1], products.iloc[::-1])
        self.assertEqual(first.recommend({"H", "N"}, 10, user_id="u1"), ["A", "B"])
        self.assertEqual(first.recommend({"H", "N"}, 10, user_id="u1"), second.recommend({"H", "N"}, 10, user_id="u1"))
        np.testing.assert_array_equal(first.user_profile("u1").toarray(), second.user_profile("u1").toarray())

    def test_unknown_missing_and_unusable_profiles_are_explicit(self):
        model = TfidfContentRecommender()
        with self.assertRaisesRegex(RuntimeError, "Fit"):
            model.recommend(set(), user_id="u1")
        model.fit(self.train, self.products)
        for user in (None, "unknown"):
            with self.subTest(user=user), self.assertRaisesRegex(ValueError, "No positive training profile"):
                model.recommend(set(), user_id=user)
        with self.assertRaisesRegex(ValueError, "k"):
            model.recommend(set(), 0, user_id="u1")
        products = self.products.copy()
        products.loc[products["product_id"].eq("H"), "product_name"] = None
        model.fit(self.train, products)
        with self.assertRaisesRegex(ValueError, "No usable metadata profile"):
            model.recommend(set(), user_id="u1")

    def test_shared_explanation_terms_and_contributions(self):
        model = TfidfContentRecommender().fit(self.train, self.products)
        terms = model.top_matching_terms("u1", "A", 100)
        self.assertTrue(terms)
        self.assertEqual(terms, model.top_matching_terms("u1", "A", 100))
        self.assertEqual(terms, sorted(terms, key=lambda pair: (-pair[1], pair[0])))
        analyzer = model.vectorizer.build_analyzer()
        shared = set(analyzer(self.products.loc[0, "product_name"])) & set(analyzer(self.products.loc[1, "product_name"]))
        self.assertEqual({term for term, _ in terms}, shared)
        score = (model.user_profile("u1") @ model.product_vectors.getrow(0).T).toarray()[0, 0]
        self.assertAlmostEqual(sum(value for _, value in terms), score)
        self.assertEqual(model.top_matching_terms("u1", "B"), [])
        with self.assertRaisesRegex(ValueError, "train candidate catalog"):
            model.top_matching_terms("u1", "FUTURE")

    def test_feature_fields_missing_values_and_no_input_mutation(self):
        products = self.products.assign(rating=999, reviews=888, loves_count=777)
        before = products.copy(deep=True)
        features = build_product_features(products)
        self.assertFalse(features["feature_text"].str.contains("999|888|777|None|<NA>").any())
        self.assertEqual(features["product_id"].tolist(), sorted(products["product_id"]))
        original_train = self.train.copy(deep=True)
        TfidfContentRecommender().fit(self.train, products)
        assert_frame_equal(products, before)
        assert_frame_equal(self.train, original_train)
        self.assertEqual(build_product_features(pd.DataFrame({"product_id": ["001"]})).loc[0, "feature_text"], "")

    def test_future_metadata_cannot_change_vectors_or_ranking(self):
        model = TfidfContentRecommender().fit(self.train, self.products)
        modified = self.products.copy()
        modified.loc[modified["product_id"].eq("FUTURE"), "product_name"] = "hyaluronic charcoal " * 100
        other = TfidfContentRecommender().fit(self.train, modified)
        np.testing.assert_array_equal(model.product_vectors.toarray(), other.product_vectors.toarray())
        self.assertEqual(model.recommend({"H", "N"}, user_id="u1"), other.recommend({"H", "N"}, user_id="u1"))

    def test_same_evaluator_cohort_and_no_test_profile_updates(self):
        train = pd.concat([self.train, pd.DataFrame({
            "author_id": ["u1"], "product_id": ["N"], "rating": [4], "positive": [1],
        })], ignore_index=True).drop_duplicates(["author_id", "product_id"], keep="last")
        test = pd.DataFrame({"author_id": ["u1", "other"], "product_id": ["A", "FUTURE"], "positive": [1, 1]})
        model = TfidfContentRecommender().fit(train, self.products)
        before = model.user_profile("u1").toarray()
        content = evaluate_ranking(model, train, test)
        popular = evaluate_ranking(MostPopularRecommender().fit(train), train, test)
        for key in ("history_eligible_users", "servable_evaluation_users", "cold_start_only_users", "candidate_train_products"):
            self.assertEqual(content[key], popular[key])
        np.testing.assert_array_equal(before, model.user_profile("u1").toarray())
