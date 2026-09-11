"""Tiny synthetic checks of collaborative signal, leakage, and diagnostics."""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from scipy.sparse import issparse

from glowguide.evaluation import evaluate_ranking
from glowguide.recommenders.collaborative import ItemItemCollaborativeRecommender
from glowguide.split import temporal_split


def frame(rows):
    return pd.DataFrame(rows, columns=["author_id", "product_id", "positive", "rating"])


@pytest.fixture
def train():
    return frame([
        ("u", "A", 1, 4), ("u", "NEG", 0, 1),
        ("v", "A", 1, 5), ("v", "B", 1, 4), ("v", "C", 1, 4),
        ("w", "A", 1, 4), ("w", "B", 1, 5),
        ("x", "Z", 1, 5), ("negative", "NEG", 0, 2),
    ])


def sim(model, a, b):
    return model.item_similarity[model.product_ids.index(a), model.product_ids.index(b)]


def test_overlap_exact_formula_unrelated_and_diagonal(train):
    model = ItemItemCollaborativeRecommender().fit(train)
    assert sim(model, "A", "B") == pytest.approx(2 / np.sqrt(3 * 2) * 2 / 12)
    assert sim(model, "A", "C") > 0
    assert sim(model, "A", "Z") == 0
    assert np.all(model.item_similarity.diagonal() == 0)
    assert np.isfinite(model.item_similarity.data).all()
    assert (model.item_similarity != model.item_similarity.T).nnz == 0
    assert issparse(model.positive_matrix) and issparse(model.item_similarity)


def test_greater_evidence_ranks_higher(train):
    model = ItemItemCollaborativeRecommender().fit(train)
    assert model.recommend({"A", "NEG"}, 2, user_id="u") == ["B", "C"]


def test_shrinkage_reduces_rare_similarity(train):
    raw = ItemItemCollaborativeRecommender(0).fit(train)
    shrunk = ItemItemCollaborativeRecommender(10).fit(train)
    assert sim(shrunk, "A", "C") == pytest.approx(sim(raw, "A", "C") / 11)
    assert sim(shrunk, "A", "C") < sim(raw, "A", "C")


def test_negatives_do_not_create_similarity_or_profile(train):
    model = ItemItemCollaborativeRecommender().fit(train)
    assert sim(model, "A", "NEG") == 0
    assert model.user_profile("negative").nnz == 0
    extra = frame([("u", "B", 0, 2)])
    after = ItemItemCollaborativeRecommender().fit(pd.concat([train, extra]))
    np.testing.assert_array_equal(model.item_similarity.toarray(), after.item_similarity.toarray())
    assert after.recommend({"A", "NEG", "B"}, 10, user_id="u") == ["C"]


def test_rating_five_contributes_twice_and_explanation_reconstructs():
    data = frame([("u", "A", 1, 4), ("u", "B", 1, 5),
                  ("v", "A", 1, 4), ("v", "C", 1, 5),
                  ("w", "B", 1, 4), ("w", "C", 1, 4)])
    model = ItemItemCollaborativeRecommender().fit(data)
    explanation = model.explain_score("u", "C")
    b, a = explanation["contributions"]
    assert b["history_product_id"] == "B"
    assert b["adjusted_similarity"] == a["adjusted_similarity"]
    assert b["weighted_contribution"] == 2 * a["weighted_contribution"]
    assert b["score_contribution"] == 2 * a["score_contribution"]
    assert explanation["final_score"] == pytest.approx(sum(x["weighted_contribution"] for x in (a, b)) / 3)
    assert explanation["final_score"] == pytest.approx(sum(x["score_contribution"] for x in (a, b)))
    short = model.explain_score("u", "C", top_n=1)
    assert short["final_score"] == pytest.approx(short["contributions"][0]["score_contribution"] + short["other_score_contribution"])


def test_seen_catalog_no_duplicates_no_filler_and_short_lists(train):
    model = ItemItemCollaborativeRecommender().fit(train)
    result = model.recommend({"A", "NEG", "B", "TEST_ONLY"}, 10, user_id="u")
    assert result == ["C"]
    assert set(result) <= set(train.product_id)
    assert len(result) == len(set(result)) < 10
    assert "Z" not in result
    assert model.recommend(model.candidate_product_ids, 10, user_id="u") == []


def test_deterministic_exact_ties():
    data = frame([("u", "A", 1, 4), ("v", "A", 1, 5),
                  ("v", "C", 1, 4), ("v", "B", 1, 4)])
    for source in (data, data.iloc[::-1], data):
        model = ItemItemCollaborativeRecommender().fit(source)
        assert sim(model, "A", "B") == sim(model, "A", "C")
        assert model.recommend({"A"}, 10, user_id="u") == ["B", "C"]


def test_unknown_empty_profiles_and_all_negative_training(train):
    model = ItemItemCollaborativeRecommender().fit(train)
    for user in (None, "unknown", "negative"):
        assert model.recommend(set(), 10, user_id=user) == []
        assert model.explain_score(user, "A")["final_score"] == 0
    model.fit(frame([("u", "A", 0, 1)]))
    assert model.candidate_product_ids == {"A"}
    assert model.item_similarity.nnz == 0
    assert model.recommend(set(), user_id="u") == []


def test_duplicate_pairs_count_unique_users(train):
    a = ItemItemCollaborativeRecommender().fit(train)
    b = ItemItemCollaborativeRecommender().fit(pd.concat([train, train]))
    np.testing.assert_array_equal(a.support, b.support)
    np.testing.assert_array_equal(a.item_similarity.toarray(), b.item_similarity.toarray())
    np.testing.assert_array_equal(a.user_profile("v").toarray(), b.user_profile("v").toarray())


def test_fit_and_recommend_do_not_mutate_inputs(train):
    before = train.copy(deep=True)
    seen = {"A", "NEG"}
    model = ItemItemCollaborativeRecommender().fit(train)
    model.recommend(seen, user_id="u")
    model.explain_score("u", "B")
    assert_frame_equal(train, before)
    assert seen == {"A", "NEG"}
    profile = model.user_profile("u")
    profile.data[:] = 0
    assert model.user_profile("u").sum() == 1


def test_added_and_changed_future_rows_cannot_change_fit(train):
    past = train.assign(submission_time="2022-02-02")
    futures = [frame([("u", "FUTURE", 1, 5)]),
               frame([("u", "B", 0, 1), ("u", "Z", 1, 5)])]
    models = []
    for future in futures:
        combined = pd.concat([past, future.assign(submission_time="2023-01-01")], ignore_index=True)
        split = temporal_split(combined, 0.8)
        assert split.cutoff == pd.Timestamp("2022-02-02")
        assert_frame_equal(split.train.reset_index(drop=True), past.assign(submission_time=pd.Timestamp("2022-02-02")))
        models.append(ItemItemCollaborativeRecommender().fit(split.train))
    a, b = models
    np.testing.assert_array_equal(a.item_similarity.toarray(), b.item_similarity.toarray())
    np.testing.assert_array_equal(a.user_profile("u").toarray(), b.user_profile("u").toarray())
    assert a.recommend({"A", "NEG"}, user_id="u") == b.recommend({"A", "NEG"}, user_id="u")
    assert "FUTURE" not in a.candidate_product_ids


def test_generic_evaluator_counts_and_fixed_precision_denominator(train):
    model = ItemItemCollaborativeRecommender().fit(train)
    test = frame([("u", "B", 1, 5), ("x", "B", 1, 4)])
    result = evaluate_ranking(model, train, test, k=2, min_train_positives=1)
    assert result["mean_recommendations_returned"] == 1
    assert result["median_recommendations_returned"] == 1
    assert result["users_with_full_k_recommendations"] == 1
    assert result["users_with_fewer_than_k_recommendations"] == 1
    assert result["users_with_zero_recommendations"] == 1
    assert result["percentage_users_with_full_k_recommendations"] == 50
    assert result["percentage_users_with_fewer_than_k_recommendations"] == 50
    assert result["percentage_users_with_zero_recommendations"] == 50
    assert result["precision_at_k"] == 0.25


def test_invalid_inputs(train):
    for value in (-1, np.nan, np.inf, True, "10"):
        with pytest.raises(ValueError, match="shrinkage"):
            ItemItemCollaborativeRecommender(value)
    model = ItemItemCollaborativeRecommender()
    with pytest.raises(RuntimeError, match="Fit"):
        model.recommend(set())
    for data in (train.iloc[:0], train.drop(columns="rating"), train.assign(positive=3), train.assign(rating=1)):
        with pytest.raises(ValueError):
            model.fit(data)
    model.fit(train)
    with pytest.raises(ValueError, match="train catalog"):
        model.explain_score("u", "FUTURE")
    with pytest.raises(ValueError):
        model.recommend(set(), 0)
