"""Offline ranking evaluation against positive future interactions."""

from collections.abc import Collection
from numbers import Integral
from time import perf_counter
from typing import Protocol

import numpy as np
import pandas as pd

from .metrics import (
    average_precision_at_k,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    validate_k,
)


class RankingRecommender(Protocol):
    candidate_product_ids: frozenset[str]

    def recommend(self, seen_product_ids: Collection[str], k: int, *, user_id: str | None = None) -> list[str]: ...


def eligible_evaluation_users(
    train: pd.DataFrame, test: pd.DataFrame, min_train_positives: int = 2
) -> pd.Index:
    """Select users with sufficient positive train rows and any positive test row.

    Inputs are processed, deduplicated interactions. Eligibility is determined
    before restricting future relevance to the train catalog.
    """
    if isinstance(min_train_positives, bool) or not isinstance(min_train_positives, Integral) or min_train_positives < 1:
        raise ValueError("min_train_positives must be a positive integer")
    counts = train.loc[train["positive"].eq(1)].groupby("author_id").size()
    future_users = test.loc[test["positive"].eq(1), "author_id"].unique()
    return counts.index[counts.ge(min_train_positives) & counts.index.isin(future_users)].sort_values()


def evaluate_ranking(
    model: RankingRecommender,
    train: pd.DataFrame,
    test: pd.DataFrame,
    k: int = 10,
    min_train_positives: int = 2,
) -> dict[str, int | float]:
    """Macro-average scores over history-eligible users with reachable relevance.

    Relevant products are positive test products in the train candidate catalog.
    History-eligible users with none remaining are cold-start-only diagnostics,
    excluded from ranking, coverage, and latency. Interaction diagnostics retain
    all history-eligible users; their positive test rows form the rate denominator.
    All train interactions form seen sets. Only recommend() is timed, excluding
    fitting, eligibility, grouping, and metric computation. AP uses min(K, |R|).
    """
    validate_k(k)
    history_users = eligible_evaluation_users(train, test, min_train_positives)
    if history_users.empty:
        raise ValueError("No eligible evaluation users for the configured positive-history threshold")
    catalog = frozenset(train["product_id"].unique())
    if model.candidate_product_ids != catalog:
        raise ValueError("Model candidate catalog must equal the products appearing in train")
    positive_test = test.loc[test["positive"].eq(1) & test["author_id"].isin(history_users)]
    reachable = positive_test.loc[positive_test["product_id"].isin(catalog)]
    relevant_by_user = reachable.groupby("author_id")["product_id"].agg(set).to_dict()
    servable_users = history_users[history_users.isin(relevant_by_user)]
    if servable_users.empty:
        raise ValueError("No servable evaluation users: history-eligible users have no positive test products in the train catalog")
    cold_start_only_users = len(history_users) - len(servable_users)
    outside_catalog_interactions = len(positive_test) - len(reachable)
    seen_by_user = (
        train.loc[train["author_id"].isin(servable_users)]
        .groupby("author_id")["product_id"].agg(set).to_dict()
    )
    metrics = {
        "precision_at_k": precision_at_k,
        "recall_at_k": recall_at_k,
        "hit_rate_at_k": hit_rate_at_k,
        "ndcg_at_k": ndcg_at_k,
        "map_at_k": average_precision_at_k,
    }
    totals = dict.fromkeys(metrics, 0.0)
    latencies = []
    recommended_catalog = set()
    for user in servable_users:
        seen = seen_by_user[user]
        relevant = relevant_by_user[user]
        start = perf_counter()
        recommended = model.recommend(seen, k, user_id=user)
        latencies.append((perf_counter() - start) * 1000)
        # Cheap runtime checks make candidate and seen-item assumptions explicit.
        returned = set(recommended)
        if len(returned) != len(recommended) or len(recommended) > k:
            raise ValueError("Recommender returned duplicates or more than K products")
        if returned & seen or not returned <= catalog:
            raise ValueError("Recommender returned seen products or products outside the train catalog")
        recommended_catalog.update(returned)
        for name, metric in metrics.items():
            totals[name] += metric(recommended, relevant, k)
    return {
        "candidate_train_products": len(catalog),
        "history_eligible_users": len(history_users),
        "servable_evaluation_users": len(servable_users),
        "cold_start_only_users": cold_start_only_users,
        "cold_start_only_user_rate": cold_start_only_users / len(history_users),
        "eligible_positive_test_interactions": len(positive_test),
        "relevant_test_interactions_used": len(reachable),
        "positive_test_interactions_outside_train_catalog": outside_catalog_interactions,
        "positive_test_interactions_outside_train_catalog_rate": outside_catalog_interactions / len(positive_test),
        **{name: total / len(servable_users) for name, total in totals.items()},
        "catalog_coverage_at_k": len(recommended_catalog) / len(catalog),
        "mean_recommendation_latency_ms": float(np.mean(latencies)),
        "p95_recommendation_latency_ms": float(np.percentile(latencies, 95)),
    }
