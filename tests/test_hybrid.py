"""Synthetic score alignment, fusion arithmetic, and explainability checks."""

import numpy as np
import pandas as pd
import pytest

from glowguide.recommenders.hybrid import COMPONENTS, HybridRecommender, normalize_scores, rank_scores
from glowguide.tuning import fit_components, weight_dict


class Scores:
    def __init__(self, values, products=("A", "B", "C")):
        self.product_ids = products
        self.candidate_product_ids = frozenset(products)
        self.values = np.array(values, dtype=float)

    def score_candidates(self, *, user_id=None):
        return self.values.copy()


def models(*vectors):
    return dict(zip(COMPONENTS, [Scores(v) for v in vectors]))


@pytest.fixture
def fitted():
    train = pd.DataFrame([
        ("u", "A", 1, 4), ("u", "B", 1, 5), ("u", "D", 0, 1),
        ("v", "A", 1, 4), ("v", "C", 1, 5), ("w", "B", 1, 4), ("w", "C", 1, 4),
    ], columns=["author_id", "product_id", "positive", "rating"]).assign(submission_time=pd.Timestamp("2021-01-01"), skin_type="dry", skin_tone="fair")
    products = pd.DataFrame({"product_id": ["A", "B", "C", "D", "FUTURE"],
                             "product_name": ["aloe cream", "aloe serum", "gentle cream", "", "future unique"]})
    return fit_components(train, products)


def test_raw_score_alignment_semantics_and_standalone_ranking(fitted):
    for name, model in fitted.items():
        assert model.product_ids == ("A", "B", "C", "D")
        scores = model.score_candidates(user_id="u")
        assert scores.shape == (4,) and np.isfinite(scores).all()
        expected = sorted(range(4), key=lambda i: (-scores[i], model.product_ids[i]))
        ranked = [model.product_ids[i] for i in expected if name != "collaborative" or scores[i] > 0]
        assert model.recommend(set(), 10, user_id="u") == ranked
        assert scores[0] >= 0  # API does not exclude the user's seen A/B.
    np.testing.assert_array_equal(fitted["popularity"].score_candidates(), [2, 2, 2, 0])
    for name in ("profile", "collaborative"):
        model = fitted[name]
        for i, product_id in enumerate(model.product_ids):
            assert model.score_candidates(user_id="u")[i] == pytest.approx(model.explain_score("u", product_id)["final_score"])
    for name in ("content", "collaborative"):
        assert not fitted[name].score_candidates(user_id="unknown").any()
    with pytest.raises(ValueError):
        fitted["content"].recommend(set(), user_id="unknown")
    copy = fitted["popularity"].score_candidates()
    copy[:] = 99
    assert fitted["popularity"].score_candidates()[0] == 2
    with pytest.raises(AttributeError):
        fitted["popularity"].product_ids = ("changed",)


def test_unusable_content_metadata_returns_zero(fitted):
    content = fitted["content"]
    # A known user's zero usable vector follows the same documented API behavior.
    content.product_vectors.data[:] = 0
    np.testing.assert_array_equal(content.score_candidates(user_id="u"), np.zeros(4))


def test_catalog_and_order_mismatch():
    components = models(*([[1, 2, 3]] * 4))
    components["content"] = Scores([1, 2, 3], ("A", "B", "X"))
    with pytest.raises(ValueError, match="catalog"):
        HybridRecommender(components, weight_dict((.25,) * 4))
    components["content"] = Scores([1, 2, 3], ("B", "A", "C"))
    with pytest.raises(ValueError, match="ordering"):
        HybridRecommender(components, weight_dict((.25,) * 4))


def test_normalization_availability_and_exact_fusion():
    components = models([2, 4, 0], [0, 0, 0], [3, 0, 6], [0, 10, 5])
    weights = weight_dict((.1, .3, .1, .5))
    hybrid = HybridRecommender(components, weights)
    raw, normalized, available = hybrid.component_scores("u")
    np.testing.assert_array_equal(raw[0], [2, 4, 0])
    np.testing.assert_array_equal(normalized, [[.5, 1, 0], [0, 0, 0], [.5, 0, 1], [0, 1, .5]])
    np.testing.assert_array_equal(available, [True, False, True, True])
    expected = (.1 * normalized[0] + .1 * normalized[2] + .5 * normalized[3]) / .7
    np.testing.assert_allclose(hybrid.score_candidates(user_id="u"), expected)
    explanation = hybrid.explain_score("u", "B")
    assert explanation["configured_weights"] == weights
    assert explanation["effective_weights"] == pytest.approx(weight_dict((1/7, 0, 1/7, 5/7)))
    assert not explanation["components"]["content"]["available"]
    assert sum(c["weighted_contribution"] for c in explanation["components"].values()) == pytest.approx(explanation["final_score"])
    assert weights == weight_dict((.1, .3, .1, .5))
    with pytest.raises(TypeError):
        hybrid.configured_weights["popularity"] = 1


def test_seen_ties_duplicates_and_zero_filler():
    hybrid = HybridRecommender(models(*([[1, 1, 0]] * 4)), weight_dict((.25,) * 4))
    assert hybrid.recommend(set(), 10) == ["A", "B"]
    assert hybrid.recommend({"A", "FUTURE"}, 10) == ["B"]
    assert hybrid.recommend({"A", "B"}, 10) == []
    assert hybrid.recommend(set(), 1) == ["A"]
    assert hybrid.recommend(set(), 10) == hybrid.recommend(set(), 10)
    # Partition cutoff ties resolve by lexical IDs, even in a nonlexical catalog.
    assert rank_scores(np.array([1, 1, 1]), ("C", "B", "A"), set(), 2) == ["A", "B"]


def test_all_positive_weight_signals_unavailable():
    hybrid = HybridRecommender(models([1, 2, 3], [0, 0, 0], [1, 1, 1], [0, 0, 0]), weight_dict((0, .5, 0, .5)))
    assert hybrid.recommend(set()) == []
    assert sum(hybrid.explain_score(None, "A")["effective_weights"].values()) == 0
    zero, available = normalize_scores(np.array([[-1, -2]] * 4))
    assert not zero.any() and not available.any()


@pytest.mark.parametrize("values", [(-.1, .1, .5, .5), (float("nan"), 0, 0, 1),
                                    (float("inf"), 0, 0, 0), (.1, .1, .1, .1), (0, 0, 0, 0)])
def test_invalid_weights(values):
    with pytest.raises(ValueError, match="Weights"):
        HybridRecommender(models(*([[1, 2, 3]] * 4)), weight_dict(values))


def test_nonfinite_or_misaligned_raw_scores_rejected():
    for values in ([np.nan, 1, 2], [np.inf, 1, 2], [1, 2]):
        components = models(*([[1, 2, 3]] * 4))
        components["content"].values = np.array(values)
        with pytest.raises(ValueError):
            HybridRecommender(components, weight_dict((.25,) * 4)).score_candidates()


def test_existing_explanations_reused_and_train_only_candidates(fitted):
    hybrid = HybridRecommender(fitted, weight_dict((.25,) * 4))
    explanation = hybrid.explain_score("u", "C")
    assert explanation["components"]["content"]["matching_terms"] == fitted["content"].top_matching_terms("u", "C")
    for name in ("profile", "collaborative"):
        assert explanation["components"][name]["explanation"] == fitted[name].explain_score("u", "C")
    assert sum(v["weighted_contribution"] for v in explanation["components"].values()) == pytest.approx(explanation["final_score"])
    assert "FUTURE" not in hybrid.candidate_product_ids
    assert set(hybrid.recommend({"A", "B", "D"}, user_id="u")) <= {"C"}
