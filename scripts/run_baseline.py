#!/usr/bin/env python3
"""Run Model 0 with a global temporal holdout and save benchmark results."""

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from glowguide.evaluation import evaluate_ranking
from glowguide.metrics import validate_k
from glowguide.recommenders import MostPopularRecommender
from glowguide.split import temporal_split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--split-quantile", type=float, default=0.80)
    parser.add_argument("--min-train-positives", type=int, default=2)
    args = parser.parse_args()
    validate_k(args.k)
    if not 0 < args.split_quantile < 1:
        parser.error("--split-quantile must be strictly between 0 and 1")
    if args.min_train_positives < 1:
        parser.error("--min-train-positives must be a positive integer")

    start = perf_counter()
    path = PROJECT_ROOT / "data" / "processed" / "interactions.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Processed interactions not found: {path}. Run build_processed_data.py first.")
    print("Loading processed interactions and splitting chronologically...", flush=True)
    interactions = pd.read_csv(
        path,
        dtype={column: "string" for column in (
            "author_id", "product_id", "skin_type", "skin_tone", "eye_color", "hair_color",
        )},
        low_memory=False,
    )
    split = temporal_split(interactions, args.split_quantile)
    print("Fitting train-only popularity and evaluating servable users...", flush=True)
    model = MostPopularRecommender().fit(split.train)
    evaluation = evaluate_ranking(model, split.train, split.test, args.k, args.min_train_positives)
    results = {
        "model": "most_popular",
        "k": args.k,
        "split_quantile": args.split_quantile,
        "temporal_cutoff": split.cutoff.isoformat(),
        "min_train_positives": args.min_train_positives,
        "train_interactions": len(split.train),
        "test_interactions": len(split.test),
        "train_positive_interactions": int(split.train["positive"].eq(1).sum()),
        "test_positive_interactions": int(split.test["positive"].eq(1).sum()),
        "train_max_timestamp": split.train["submission_time"].max().isoformat(),
        "test_min_timestamp": split.test["submission_time"].min().isoformat(),
        "relevance_policy": "positive test products in train catalog; primary metrics, coverage, and latency use servable history-eligible users only",
        "diagnostic_policy": "cold-start user rate uses all history-eligible users; outside-catalog interaction rate uses all their positive test interactions",
        "ap_denominator": "min(k, number of in-catalog relevant products)",
        "candidate_policy": "all train products; no inventory filtering",
        **evaluation,
        "experiment_runtime_seconds": perf_counter() - start,
    }
    print("\nGlowGuide Model 0: Most Popular")
    print(f"Temporal cutoff: {split.cutoff}")
    for key in (
        "train_interactions", "test_interactions", "train_positive_interactions",
        "test_positive_interactions", "candidate_train_products",
    ):
        print(f"{key.replace('_', ' ').capitalize()}: {results[key]:,}")
    print(f"History eligible users: {results['history_eligible_users']:,}")
    print(f"Servable evaluation users: {results['servable_evaluation_users']:,}")
    print(f"Cold-start-only users: {results['cold_start_only_users']:,}")
    print(f"Cold-start-only user rate: {results['cold_start_only_user_rate']:.6%}")
    print(f"Eligible positive test interactions: {results['eligible_positive_test_interactions']:,}")
    print(f"Relevant in-catalog test interactions: {results['relevant_test_interactions_used']:,}")
    print(f"Positive test interactions outside train catalog: {results['positive_test_interactions_outside_train_catalog']:,}")
    print(f"Outside-train-catalog positive interaction rate: {results['positive_test_interactions_outside_train_catalog_rate']:.6%}")
    print("Primary ranking metrics, coverage, and latency: servable cohort only.")
    for label, key in (
        ("Precision", "precision_at_k"), ("Recall", "recall_at_k"),
        ("HitRate", "hit_rate_at_k"), ("NDCG", "ndcg_at_k"),
        ("MAP", "map_at_k"), ("Catalog Coverage", "catalog_coverage_at_k"),
    ):
        print(f"{label}@{args.k}: {results[key]:.12f}")
    print(f"Mean recommendation latency: {results['mean_recommendation_latency_ms']:.6f} ms")
    print(f"P95 recommendation latency: {results['p95_recommendation_latency_ms']:.6f} ms")
    print(f"Experiment runtime (load through evaluation): {results['experiment_runtime_seconds']:.3f} s")
    output = PROJECT_ROOT / "artifacts" / "baseline_metrics.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("Saved artifacts/baseline_metrics.json")


if __name__ == "__main__":
    main()
