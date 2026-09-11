"""Nested temporal fusion-weight selection. This module accepts no OUTER TEST.

Float64 normalized scores are cached on disk once per inner servable user.
Search streams one user's four catalog vectors at a time; neither fitting nor
component scoring is repeated per weight. Temporary mappings are closed before
directory cleanup on Windows. All objective metrics reuse existing functions.
"""

from functools import cmp_to_key
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np

from .evaluation import eligible_evaluation_users, evaluate_ranking
from .metrics import ndcg_at_k, average_precision_at_k, recall_at_k, validate_k
from .recommenders.popularity import MostPopularRecommender
from .recommenders.content import TfidfContentRecommender
from .recommenders.profile import SkinProfileRecommender
from .recommenders.collaborative import ItemItemCollaborativeRecommender
from .recommenders.hybrid import COMPONENTS, HybridRecommender, fuse_scores, rank_unseen_scores, validate_weights
from .split import temporal_split


OBJECTIVE_METRICS = {"ndcg_at_k": ndcg_at_k, "map_at_k": average_precision_at_k, "recall_at_k": recall_at_k}
TOLERANCE = 1e-12
SELECTION_POLICY = "maximize inner NDCG@K, then MAP@K, then Recall@K; absolute tolerance 1e-12; then lexicographically smallest (popularity, content, profile, collaborative)"
INTEGRITY_STATEMENT = "Fusion weights were selected using a nested global temporal validation split inside the outer training period. The outer test set was not used for fusion-weight selection."
FROZEN_STATEMENT = "The outer test was evaluated only after fusion weights were frozen."


def weight_dict(values):
    return dict(zip(COMPONENTS, map(float, values)))


def coarse_grid() -> list[tuple[float, ...]]:
    """All 35 integer-lattice simplex points at resolution 1/4."""
    return [tuple(v / 4 for v in values) for values in product(range(5), repeat=4) if sum(values) == 4]


def local_grid(center) -> list[tuple[float, ...]]:
    """Resolution 1/20, each coordinate within inclusive +/- 1/4 of center."""
    validate_weights(weight_dict(center))
    units = np.rint(np.asarray(center) * 20).astype(int)
    ranges = [range(max(0, v - 5), min(20, v + 5) + 1) for v in units]
    return sorted({tuple(v / 20 for v in values) for values in product(*ranges) if sum(values) == 20})


def compare_results(a: dict, b: dict) -> int:
    """Negative means a wins. Near-equal metrics advance to the next key."""
    for metric in OBJECTIVE_METRICS:
        delta = a["metrics"][metric] - b["metrics"][metric]
        if abs(delta) > TOLERANCE:
            return -1 if delta > 0 else 1
    aw = tuple(a["weights"][name] for name in COMPONENTS)
    bw = tuple(b["weights"][name] for name in COMPONENTS)
    return (aw > bw) - (aw < bw)


def sort_results(results: list[dict]) -> list[dict]:
    # Start in weight order to make even tolerance-boundary cases repeatable.
    ordered = sorted(results, key=lambda r: tuple(r["weights"][n] for n in COMPONENTS))
    return sorted(ordered, key=cmp_to_key(compare_results))


def fit_components(train, products, shrinkage=10.0) -> dict:
    """Each component receives only this explicit training frame/catalog."""
    return {
        "popularity": MostPopularRecommender().fit(train),
        "content": TfidfContentRecommender().fit(train, products),
        "profile": SkinProfileRecommender().fit(train),
        "collaborative": ItemItemCollaborativeRecommender(shrinkage).fit(train),
    }


class InnerScoreCache:
    """Reusable inner-only cohort, relevance, seen indices and float64 scores."""

    def __init__(self, components, train, validation, path: Path, min_train_positives=2):
        self.model = HybridRecommender(components, weight_dict((.25,) * 4))
        self.product_ids = self.model.product_ids
        self.candidate_product_ids = self.model.candidate_product_ids
        if self.candidate_product_ids != frozenset(train.product_id):
            raise ValueError("Inner component catalog must match INNER TRAIN")
        history = eligible_evaluation_users(train, validation, min_train_positives)
        positive = validation.loc[validation.positive.eq(1) & validation.author_id.isin(history)]
        reachable = positive.loc[positive.product_id.isin(self.candidate_product_ids)]
        relevant = reachable.groupby("author_id").product_id.agg(set).to_dict()
        self.users = tuple(history[history.isin(relevant)])
        if not self.users:
            raise ValueError("No inner servable validation users")
        seen = train.loc[train.author_id.isin(self.users)].groupby("author_id").product_id.agg(set).to_dict()
        index = {p: i for i, p in enumerate(self.product_ids)}
        self.seen = [[index[p] for p in seen[u]] for u in self.users]
        self.relevant = [relevant[u] for u in self.users]
        self._user_index = {u: i for i, u in enumerate(self.users)}
        self.metadata = {"inner_candidate_products": len(index), "inner_history_eligible_users": len(history),
                         "inner_servable_users": len(self.users), "inner_cold_start_only_users": len(history) - len(self.users),
                         "inner_positive_validation_interactions_outside_catalog": len(positive) - len(reachable)}
        self.scores = np.memmap(path, mode="w+", dtype="float64", shape=(len(self.users), 4, len(index)))
        self.available = np.zeros((len(self.users), 4), dtype=bool)
        try:
            for i, user in enumerate(self.users):
                _, self.scores[i], self.available[i] = self.model.component_scores(user)
            self.scores.flush()
        except BaseException:
            self.close()
            raise
        self.weights = np.array([.25] * 4)

    def close(self):
        self.scores._mmap.close()

    def recommend(self, seen_product_ids, k=10, *, user_id=None):
        i = self._user_index[user_id]
        scores, _ = fuse_scores(self.scores[i], self.available[i], self.weights)
        # The shared evaluator supplies the same TRAIN seen set retained here.
        return rank_unseen_scores(scores, self.product_ids, self.seen[i], k)

    def objective(self, weights, k, *, empty_ablation=False):
        # Removing the only selected component leaves no weights to normalize.
        # Evaluate an empty recommender on the same cohort for that diagnostic.
        values = np.zeros(4) if empty_ablation and all(weights[n] == 0 for n in COMPONENTS) else validate_weights(weights)
        totals = dict.fromkeys(OBJECTIVE_METRICS, 0.0)
        for i in range(len(self.users)):
            scores, _ = fuse_scores(self.scores[i], self.available[i], values)
            ranked = rank_unseen_scores(scores, self.product_ids, self.seen[i], k)
            for name, metric in OBJECTIVE_METRICS.items():
                totals[name] += metric(ranked, self.relevant[i], k)
        return {key: value / len(self.users) for key, value in totals.items()}


def tune_hybrid(outer_train, products, *, inner_split_quantile=.8, k=10, min_train_positives=2,
                collaborative_shrinkage=10.0, progress=print) -> dict:
    """Select and diagnose weights using OUTER TRAIN only; never refit on it.

    Callers must freeze this result before fitting fresh outer components.
    Ablations are diagnostic and cannot modify the chosen configuration.
    """
    validate_k(k)
    start = perf_counter()
    inner = temporal_split(outer_train, inner_split_quantile)
    metadata = {"outer_train_cutoff_context": outer_train.submission_time.max().isoformat(),
                "inner_temporal_cutoff": inner.cutoff.isoformat(), "inner_split_quantile": inner_split_quantile,
                "inner_train_interactions": len(inner.train), "inner_validation_interactions": len(inner.test),
                "inner_train_max_timestamp": inner.train.submission_time.max().isoformat(),
                "inner_validation_min_timestamp": inner.test.submission_time.min().isoformat()}
    progress(f"INNER split: {metadata}")
    components = fit_components(inner.train, products, collaborative_shrinkage)
    with TemporaryDirectory(prefix="glowguide-inner-") as directory:
        cache = InnerScoreCache(components, inner.train, inner.test, Path(directory) / "scores.dat", min_train_positives)
        try:
            metadata.update(cache.metadata)
            progress(f"Inner tuning cohort: {cache.metadata}")
            evaluated = {}

            def search(grid, stage):
                results = []
                for i, weights in enumerate(grid):
                    if weights not in evaluated:
                        evaluated[weights] = {"weights": weight_dict(weights), "metrics": cache.objective(weight_dict(weights), k)}
                    results.append(evaluated[weights])
                    if (i + 1) % 10 == 0 or i + 1 == len(grid):
                        progress(f"{stage}: {i + 1}/{len(grid)} configurations evaluated")
                return sort_results(results)

            coarse = search(coarse_grid(), "Coarse search")
            progress(f"Coarse search winner: {coarse[0]}")
            local = search(local_grid(tuple(coarse[0]["weights"][n] for n in COMPONENTS)), "Local refinement")
            winner = local[0]
            selected = dict(winner["weights"])
            progress(f"Refined search winner: {winner}")
            progress(f"FINAL FROZEN WEIGHTS: {selected}")
            cache.weights = validate_weights(selected)
            full_metrics = evaluate_ranking(cache, inner.train, inner.test, k, min_train_positives)
            # Cache latency is not end-to-end recommender latency; omit it.
            full_metrics = {key: value for key, value in full_metrics.items() if "latency" not in key}
            for name in OBJECTIVE_METRICS:
                if full_metrics[name] != winner["metrics"][name]:
                    raise AssertionError("Cached search and shared evaluator metrics disagree")
            progress(f"Inner selected hybrid metrics: {full_metrics}")
            ablations = {}
            for name in COMPONENTS:
                if selected[name] == 0:
                    ablations[name] = {"applicable": False, "reason": "selected weight is zero"}
                else:
                    remaining = {n: (0.0 if n == name or selected[name] == 1 else selected[n] / (1 - selected[name])) for n in COMPONENTS}
                    metrics = cache.objective(remaining, k, empty_ablation=selected[name] == 1)
                    ablations[name] = {"applicable": True, "weights": remaining, "metrics": metrics,
                                       "change_from_full": {m: metrics[m] - full_metrics[m] for m in OBJECTIVE_METRICS}}
                    if selected[name] == 1:
                        ablations[name]["note"] = "Removing the sole selected component leaves no signal; empty recommendations evaluated on the same cohort"
            progress(f"Inner ablations (diagnostic only): {ablations}")
            return {**metadata, "k": k, "min_train_positives": min_train_positives,
                    "collaborative_shrinkage": collaborative_shrinkage,
                    "coarse_grid_size": len(coarse), "coarse_step": .25, "coarse_results": coarse, "coarse_winner": coarse[0],
                    "local_grid_size": len(local), "local_step": .05, "local_radius": .25,
                    "local_results": local, "refined_winner": winner, "selected_weights": selected,
                    "selected_on": "inner temporal validation", "selection_policy": SELECTION_POLICY,
                    "metric_tolerance": TOLERANCE, "component_order": COMPONENTS,
                    "inner_selected_metrics": full_metrics, "inner_ablations": ablations,
                    "integrity_statement": INTEGRITY_STATEMENT,
                    "cache_policy": "temporary disk-backed float64 normalized inner scores; one user at a time; deleted after tuning",
                    "tuning_runtime_seconds": perf_counter() - start}
        finally:
            cache.close()
