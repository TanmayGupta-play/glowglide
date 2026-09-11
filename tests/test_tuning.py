"""Synthetic simplex, selection, nested leakage, and cached evaluation tests."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from glowguide.evaluation import evaluate_ranking
from glowguide.recommenders.hybrid import COMPONENTS, HybridRecommender
from glowguide.split import temporal_split
from glowguide.tuning import (coarse_grid, local_grid, sort_results, weight_dict, fit_components,
                              InnerScoreCache, tune_hybrid, OBJECTIVE_METRICS)


def synthetic():
    rows = []
    for i in range(10):
        for p in (("A", "B") if i % 2 == 0 else ("A", "C")):
            rows.append((f"u{i}", p, 1, 5, "2020-01-01"))
    rows += [("u0", "C", 1, 4, "2021-01-01"), ("u1", "B", 1, 5, "2021-01-01"),
             ("u2", "C", 1, 5, "2021-01-01"), ("u3", "B", 1, 4, "2021-01-01")]
    past = pd.DataFrame(rows, columns=["author_id", "product_id", "positive", "rating", "submission_time"])
    products = pd.DataFrame({"product_id": ["A", "B", "C", "FUTURE"],
                             "product_name": ["aloe cream", "aloe serum", "gentle cream", "future"]})
    return past, products


def test_coarse_grid_is_full_deterministic_simplex():
    grid = coarse_grid()
    assert len(grid) == 35 and grid == sorted(set(grid)) == coarse_grid()
    for weights in grid:
        assert sum(weights) == 1
        assert all(w * 4 == int(w * 4) for w in weights)
    for i in range(4):
        assert tuple(float(j == i) for j in range(4)) in grid


def test_local_grid_simplex_bounds_and_determinism():
    for center in coarse_grid():
        grid = local_grid(center)
        assert grid == sorted(set(grid)) == local_grid(center)
        assert center in grid
        for weights in grid:
            assert sum(weights) == pytest.approx(1)
            assert all(0 <= w <= 1 and abs(w - c) <= .25 + 1e-12 for w, c in zip(weights, center))
            assert all(w * 20 == pytest.approx(round(w * 20)) for w in weights)


def result(weights, ndcg, ap, recall):
    return {"weights": weight_dict(weights), "metrics": dict(zip(OBJECTIVE_METRICS, (ndcg, ap, recall)))}


@pytest.mark.parametrize("a,b", [
    ((.2, .1, .1), (.1, .9, .9)),
    ((.2, .3, .1), (.2, .2, .9)),
    ((.2, .3, .4), (.2, .3, .3)),
    ((.2, .3, .4), (.2 + 1e-13, .2, .9)),
])
def test_objective_ndcg_then_map_then_recall_with_tolerance(a, b):
    winner = result((1, 0, 0, 0), *a)
    loser = result((0, 0, 0, 1), *b)
    assert sort_results([loser, winner])[0] == winner


def test_metric_ties_use_lexical_weight_order():
    results = [result(w, .2, .3, .4) for w in coarse_grid()[::-1]]
    assert sort_results(results)[0]["weights"] == weight_dict((0, 0, 0, 1))


def test_cached_objectives_equal_shared_evaluator(tmp_path):
    outer_train, products = synthetic()
    inner = temporal_split(outer_train)
    components = fit_components(inner.train, products)
    cache = InnerScoreCache(components, inner.train, inner.test, tmp_path / "scores.dat")
    try:
        for weights in (weight_dict((.25,) * 4), weight_dict((0, .5, 0, .5)), weight_dict((0, 0, 0, 1))):
            cached = cache.objective(weights, 10)
            actual = evaluate_ranking(HybridRecommender(components, weights), inner.train, inner.test)
            assert cached == {key: actual[key] for key in OBJECTIVE_METRICS}
        assert cache.objective(weight_dict((0, 0, 0, 0)), 10, empty_ablation=True) == dict.fromkeys(OBJECTIVE_METRICS, 0.0)
    finally:
        cache.close()


def test_nested_tuning_fits_only_inner_train_and_outer_test_cannot_select_weights():
    past, products = synthetic()
    original = past.copy(deep=True)
    future = pd.DataFrame([("u0", "FUTURE", 1, 5, "2022-01-01"), ("u1", "C", 0, 1, "2022-01-01")], columns=past.columns)
    changed = future.copy()
    changed.loc[0, ["product_id", "rating", "positive"]] = ["B", 1, 0]
    added = pd.concat([changed, future.iloc[:1].assign(product_id="NEW_FUTURE")], ignore_index=True)
    runs = []
    for test in (future, changed, added):
        outer = temporal_split(pd.concat([past, test], ignore_index=True))
        assert outer.cutoff == pd.Timestamp("2021-01-01")
        expected_inner = temporal_split(outer.train)
        with patch("glowguide.tuning.fit_components", wraps=fit_components) as fit:
            tuning = tune_hybrid(outer.train, products, progress=lambda _: None)
            assert fit.call_count == 1
            assert_frame_equal(fit.call_args.args[0], expected_inner.train)
        assert tuning["inner_train_interactions"] == 20
        assert tuning["inner_validation_interactions"] == 4
        assert tuning["inner_candidate_products"] == 3
        assert tuning["inner_history_eligible_users"] == tuning["inner_servable_users"] == 4
        assert tuning["selected_on"] == "inner temporal validation"
        assert not any("outer" in key and "metrics" in key for key in tuning)
        runs.append(tuning)
        final = HybridRecommender(fit_components(outer.train, products), tuning["selected_weights"])
        assert final.candidate_product_ids == frozenset(outer.train.product_id)
        assert "NEW_FUTURE" not in final.candidate_product_ids
    for key in ("selected_weights", "coarse_winner", "refined_winner", "inner_selected_metrics", "inner_ablations"):
        assert runs[0][key] == runs[1][key] == runs[2][key]
    assert_frame_equal(past, original)
