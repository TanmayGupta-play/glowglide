#!/usr/bin/env python3
"""Tune Model 4 on nested temporal validation, freeze, then evaluate OUTER TEST."""

import argparse
import json
import math
from pathlib import Path
import sys
from time import perf_counter

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from glowguide.evaluation import evaluate_ranking
from glowguide.metrics import validate_k
from glowguide.recommenders.hybrid import HybridRecommender, NORMALIZATION_POLICY, UNAVAILABLE_POLICY
from glowguide.split import temporal_split
from glowguide.tuning import tune_hybrid, fit_components, INTEGRITY_STATEMENT, FROZEN_STATEMENT


METRICS = {"precision_at_k": "Precision", "recall_at_k": "Recall", "hit_rate_at_k": "HitRate",
           "ndcg_at_k": "NDCG", "map_at_k": "MAP", "catalog_coverage_at_k": "Coverage"}
COMPONENT_MODELS = {"popularity": "most_popular", "content": "tfidf_content",
                    "profile": "skin_profile_affinity", "collaborative": "item_item_collaborative"}


def save_json(path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def comparisons(results):
    """Read frozen outer results only AFTER weight selection and final evaluation."""
    compared = {}
    keys = ("k", "split_quantile", "temporal_cutoff", "min_train_positives", "train_interactions",
            "test_interactions", "candidate_train_products", "history_eligible_users", "servable_evaluation_users",
            "cold_start_only_users", "eligible_positive_test_interactions", "relevant_test_interactions_used",
            "positive_test_interactions_outside_train_catalog", "candidate_policy", "relevance_policy", "ap_denominator")
    for model_name, filename in (("most_popular", "baseline"), ("tfidf_content", "content"),
                                 ("skin_profile_affinity", "profile"), ("item_item_collaborative", "collaborative")):
        path = PROJECT_ROOT / "artifacts" / f"{filename}_metrics.json"
        if not path.is_file():
            print(f"Optional comparison artifact absent: {path.name}", flush=True)
            continue
        baseline = json.loads(path.read_text(encoding="utf-8"))
        if baseline.get("model") != model_name or any(baseline.get(key) != results[key] for key in keys):
            print(f"Comparison omitted: {path.name} benchmark differs", flush=True)
            continue
        compared[model_name] = {key: {"baseline": baseline[key], "hybrid": results[key],
                                    "absolute_change": results[key] - baseline[key],
                                    "percentage_change": (results[key] / baseline[key] - 1) * 100 if baseline[key] else None}
                                for key in METRICS}
    return compared


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--outer-split-quantile", type=float, default=.80)
    parser.add_argument("--inner-split-quantile", type=float, default=.80)
    parser.add_argument("--min-train-positives", type=int, default=2)
    parser.add_argument("--collaborative-shrinkage", type=float, default=10)
    args = parser.parse_args()
    validate_k(args.k)
    for value in (args.outer_split_quantile, args.inner_split_quantile):
        if not 0 < value < 1:
            parser.error("Split quantiles must be strictly between 0 and 1")
    if args.min_train_positives < 1:
        parser.error("--min-train-positives must be positive")
    if not math.isfinite(args.collaborative_shrinkage) or args.collaborative_shrinkage < 0:
        parser.error("--collaborative-shrinkage must be finite and nonnegative")
    start = perf_counter()
    processed = PROJECT_ROOT / "data" / "processed"
    interactions = pd.read_csv(processed / "interactions.csv", dtype={name: "string" for name in (
        "author_id", "product_id", "skin_type", "skin_tone", "eye_color", "hair_color")}, low_memory=False)
    products = pd.read_csv(processed / "products_skincare.csv", dtype={"product_id": "string"}, low_memory=False)
    outer = temporal_split(interactions, args.outer_split_quantile)
    print(f"OUTER split: cutoff={outer.cutoff}; TRAIN={len(outer.train):,}; TEST={len(outer.test):,}; train candidates={outer.train.product_id.nunique():,}", flush=True)
    tuning = tune_hybrid(outer.train, products, inner_split_quantile=args.inner_split_quantile,
                         k=args.k, min_train_positives=args.min_train_positives,
                         collaborative_shrinkage=args.collaborative_shrinkage,
                         progress=lambda message: print(message, flush=True))
    # Persist the complete inner search and immutable selection BEFORE any outer refit/evaluation.
    save_json(PROJECT_ROOT / "artifacts" / "hybrid_tuning.json", tuning)
    frozen_weights = dict(tuning["selected_weights"])
    print(INTEGRITY_STATEMENT, flush=True)
    print("Hybrid weights are now frozen. Evaluating untouched outer test.", flush=True)
    final_start = perf_counter()
    components = fit_components(outer.train, products, args.collaborative_shrinkage)
    hybrid = HybridRecommender(components, frozen_weights)
    evaluation = evaluate_ranking(hybrid, outer.train, outer.test, args.k, args.min_train_positives)
    final_runtime = perf_counter() - final_start
    results = {
        "model": "validation_tuned_hybrid", "component_models": COMPONENT_MODELS,
        "selected_weights": frozen_weights, "selected_on": "inner temporal validation",
        "normalization_policy": NORMALIZATION_POLICY, "unavailable_component_policy": UNAVAILABLE_POLICY,
        "zero_signal_policy": "only positive final scores; no zero-score filler",
        "candidate_policy": "all train products; no inventory filtering",
        "relevance_policy": "positive test products in train catalog; primary metrics, coverage, and latency use servable history-eligible users only",
        "ap_denominator": "min(k, number of in-catalog relevant products)",
        "seen_policy": "exclude all supplied TRAIN seen products, positive and negative",
        "k": args.k, "split_quantile": args.outer_split_quantile, "temporal_cutoff": outer.cutoff.isoformat(),
        "min_train_positives": args.min_train_positives, "collaborative_shrinkage": args.collaborative_shrinkage,
        "train_interactions": len(outer.train), "test_interactions": len(outer.test),
        "train_positive_interactions": int(outer.train.positive.eq(1).sum()),
        "test_positive_interactions": int(outer.test.positive.eq(1).sum()),
        "train_max_timestamp": outer.train.submission_time.max().isoformat(),
        "test_min_timestamp": outer.test.submission_time.min().isoformat(),
        "inner_split_metadata": {key: value for key, value in tuning.items() if key.startswith("inner_") and key not in ("inner_selected_metrics", "inner_ablations")},
        "tuning_objective": tuning["selection_policy"],
        "search_metadata": {key: tuning[key] for key in ("coarse_grid_size", "coarse_step", "coarse_winner", "local_grid_size", "local_step", "local_radius", "refined_winner", "metric_tolerance", "component_order")},
        "inner_selected_metrics": tuning["inner_selected_metrics"], "inner_ablations": tuning["inner_ablations"],
        "integrity_statement": INTEGRITY_STATEMENT, "outer_evaluation_statement": FROZEN_STATEMENT,
        "tuning_runtime_seconds": tuning["tuning_runtime_seconds"], "final_refit_evaluation_runtime_seconds": final_runtime,
        **evaluation, "experiment_runtime_seconds": perf_counter() - start,
    }
    results["comparisons"] = comparisons(results)
    print(FROZEN_STATEMENT, flush=True)
    for key, label in METRICS.items():
        print(f"{label}@{args.k}: {results[key]:.17g}", flush=True)
    for key in evaluation:
        if key not in METRICS:
            print(f"{key}: {evaluation[key]}", flush=True)
    print(f"Tuning runtime: {results['tuning_runtime_seconds']:.3f} s", flush=True)
    print(f"Final refit/evaluation runtime: {final_runtime:.3f} s", flush=True)
    print(f"Total experiment runtime: {results['experiment_runtime_seconds']:.3f} s", flush=True)
    save_json(PROJECT_ROOT / "artifacts" / "hybrid_metrics.json", results)
    print("Saved artifacts/hybrid_tuning.json and artifacts/hybrid_metrics.json", flush=True)


if __name__ == "__main__":
    main()
