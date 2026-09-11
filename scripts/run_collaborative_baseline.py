#!/usr/bin/env python3
"""Evaluate Model 3 under the frozen temporal and servable-cohort benchmark."""

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
from glowguide.recommenders.collaborative import ItemItemCollaborativeRecommender
from glowguide.split import temporal_split


METRICS = {
    "precision_at_k": "Precision", "recall_at_k": "Recall", "hit_rate_at_k": "HitRate",
    "ndcg_at_k": "NDCG", "map_at_k": "MAP", "catalog_coverage_at_k": "Coverage",
}


def compare_artifact(results: dict, path: Path, model_name: str) -> dict:
    """Report comparisons only when benchmark configuration and cohorts match."""
    if not path.is_file():
        print(f"Optional comparison artifact absent: {path.name}")
        return {}
    baseline = json.loads(path.read_text(encoding="utf-8"))
    keys = [
        "k", "split_quantile", "temporal_cutoff", "min_train_positives", "train_interactions",
        "test_interactions", "candidate_train_products", "history_eligible_users", "servable_evaluation_users",
        "cold_start_only_users", "eligible_positive_test_interactions", "relevant_test_interactions_used",
        "positive_test_interactions_outside_train_catalog", "candidate_policy", "relevance_policy", "ap_denominator",
    ]
    if baseline.get("model") != model_name or any(baseline.get(key) != results[key] for key in keys):
        print(f"Comparison omitted for {path.name}: benchmark settings or cohorts differ.")
        return {}
    return {key: {
        "baseline": baseline[key], "model_3": results[key], "absolute_change": results[key] - baseline[key],
        "percentage_change": (results[key] / baseline[key] - 1) * 100 if baseline[key] else None,
    } for key in METRICS}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--split-quantile", type=float, default=0.80)
    parser.add_argument("--min-train-positives", type=int, default=2)
    parser.add_argument("--shrinkage", type=float, default=10.0)
    args = parser.parse_args()
    if not math.isfinite(args.shrinkage) or args.shrinkage < 0:
        parser.error("--shrinkage must be finite and nonnegative")
    validate_k(args.k)
    if not 0 < args.split_quantile < 1:
        parser.error("--split-quantile must be strictly between 0 and 1")
    if args.min_train_positives < 1:
        parser.error("--min-train-positives must be positive")
    start = perf_counter()
    path = PROJECT_ROOT / "data" / "processed" / "interactions.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Processed interactions missing: {path}")
    print("Loading processed interactions and applying the global temporal split...", flush=True)
    interactions = pd.read_csv(path, dtype={column: "string" for column in (
        "author_id", "product_id", "skin_type", "skin_tone", "eye_color", "hair_color",
    )}, low_memory=False)
    split = temporal_split(interactions, args.split_quantile)
    print("Fitting sparse positive TRAIN user-item cosine similarities...", flush=True)
    model = ItemItemCollaborativeRecommender(shrinkage=args.shrinkage).fit(split.train)
    print("Evaluating the unchanged servable cohort...", flush=True)
    evaluation = evaluate_ranking(model, split.train, split.test, args.k, args.min_train_positives)
    results = {
        "model": "item_item_collaborative", "k": args.k, "split_quantile": args.split_quantile,
        "temporal_cutoff": split.cutoff.isoformat(), "min_train_positives": args.min_train_positives,
        "similarity": "positive-user cosine with significance shrinkage",
        "similarity_formula": "co_count / sqrt(support_i * support_j) * co_count / (co_count + shrinkage); diagonal zero",
        "shrinkage": args.shrinkage,
        "shrinkage_policy": "fixed modeling assumption; not tuned on outer TEST",
        "rating_weighting_policy": "positive TRAIN only; rating 4 = 1.0, rating 5 = 2.0; duplicate positive pairs use maximum rating",
        "user_scoring_formula": "sum(weight_i * adjusted_similarity_i_c) / sum(weight_i)",
        "zero_signal_policy": "only scores > 0; no filler or fallback; unknown/no-positive users return []",
        "fit_policy": "TRAIN only; binary unique positive user-product pairs for similarity",
        "seen_policy": "exclude all supplied TRAIN seen items, positive and negative",
        "candidate_policy": "all train products; no inventory filtering",
        "relevance_policy": "positive test products in train catalog; primary metrics, coverage, and latency use servable history-eligible users only",
        "ap_denominator": "min(k, number of in-catalog relevant products)",
        "train_interactions": len(split.train), "test_interactions": len(split.test),
        "train_positive_interactions": int(split.train["positive"].sum()),
        "test_positive_interactions": int(split.test["positive"].sum()),
        "train_max_timestamp": split.train["submission_time"].max().isoformat(),
        "test_min_timestamp": split.test["submission_time"].min().isoformat(),
        **evaluation,
        "experiment_runtime_seconds": perf_counter() - start,
    }
    results["comparisons"] = {}
    for name, filename in (("most_popular", "baseline_metrics.json"), ("tfidf_content", "content_metrics.json"), ("skin_profile_affinity", "profile_metrics.json")):
        comparison = compare_artifact(results, PROJECT_ROOT / "artifacts" / filename, name)
        if comparison:
            results["comparisons"][name] = comparison
    print("\nGlowGuide Model 3: Item-Item Collaborative Filtering")
    print(f"Temporal cutoff: {split.cutoff}")
    for key in (
        "train_interactions", "test_interactions", "candidate_train_products", "history_eligible_users",
        "servable_evaluation_users", "cold_start_only_users", "eligible_positive_test_interactions",
        "relevant_test_interactions_used", "positive_test_interactions_outside_train_catalog",
    ):
        print(f"{key.replace('_', ' ').capitalize()}: {results[key]:,}")
    for key in ("cold_start_only_user_rate", "positive_test_interactions_outside_train_catalog_rate"):
        print(f"{key.replace('_', ' ').capitalize()}: {results[key]:.6%}")
    for key in ("mean_recommendations_returned", "median_recommendations_returned",
                "users_with_full_k_recommendations", "users_with_fewer_than_k_recommendations",
                "users_with_zero_recommendations", "percentage_users_with_full_k_recommendations",
                "percentage_users_with_fewer_than_k_recommendations", "percentage_users_with_zero_recommendations"):
        print(f"{key.replace('_', ' ').capitalize()}: {results[key]}")
    for key, label in METRICS.items():
        print(f"{label}@{args.k}: {results[key]:.17g}")
        for name, comparison in results["comparisons"].items():
            print(f"  versus {name}: {comparison[key]['absolute_change']:+.12f} absolute; {comparison[key]['percentage_change']:+.4f}%" if comparison[key]['percentage_change'] is not None else f"  versus {name}: baseline is zero")
    print(f"Mean recommendation latency: {results['mean_recommendation_latency_ms']:.6f} ms")
    print(f"P95 recommendation latency: {results['p95_recommendation_latency_ms']:.6f} ms")
    print(f"Experiment runtime (load through evaluation): {results['experiment_runtime_seconds']:.3f} s")
    output = PROJECT_ROOT / "artifacts" / "collaborative_metrics.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("Saved artifacts/collaborative_metrics.json")


if __name__ == "__main__":
    main()
