#!/usr/bin/env python3
"""Evaluate personalized TF-IDF under the Model 0 temporal benchmark."""

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import pandas as pd
import sklearn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from glowguide.evaluation import evaluate_ranking
from glowguide.features import CONTENT_FIELDS
from glowguide.metrics import validate_k
from glowguide.recommenders.content import TfidfContentRecommender, VECTORIZER_CONFIG
from glowguide.split import temporal_split


METRICS = {
    "precision_at_k": "Precision", "recall_at_k": "Recall",
    "hit_rate_at_k": "HitRate", "ndcg_at_k": "NDCG", "map_at_k": "MAP",
    "catalog_coverage_at_k": "Catalog Coverage",
}


def compare_baseline(results: dict, path: Path) -> dict:
    """Compare only artifacts with matching benchmark settings and cohorts."""
    if not path.is_file():
        print("Model 0 artifact not found; running without comparison.")
        return {}
    baseline = json.loads(path.read_text(encoding="utf-8"))
    keys = [
        "k", "split_quantile", "temporal_cutoff", "min_train_positives",
        "train_interactions", "test_interactions", "candidate_train_products",
        "history_eligible_users", "servable_evaluation_users", "cold_start_only_users",
        "eligible_positive_test_interactions", "relevant_test_interactions_used",
        "positive_test_interactions_outside_train_catalog", "relevance_policy", "ap_denominator",
    ]
    if baseline.get("model") != "most_popular" or any(baseline.get(key) != results[key] for key in keys):
        print("Model 0 artifact uses a different benchmark; comparison omitted. Rerun run_baseline.py with matching arguments.")
        return {}
    return {
        key: {
            "model_0": baseline[key], "model_1": results[key],
            "absolute_change": results[key] - baseline[key],
            "percentage_change": (results[key] - baseline[key]) / baseline[key] * 100 if baseline[key] else None,
        }
        for key in METRICS
    }


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
    processed = PROJECT_ROOT / "data" / "processed"
    for name in ("interactions.csv", "products_skincare.csv"):
        if not (processed / name).is_file():
            raise FileNotFoundError(f"Processed file missing: {processed / name}. Run build_processed_data.py first.")
    print("Loading processed data and reusing the global temporal split...", flush=True)
    interactions = pd.read_csv(processed / "interactions.csv", dtype={
        column: "string" for column in ("author_id", "product_id", "skin_type", "skin_tone", "eye_color", "hair_color")
    }, low_memory=False)
    products = pd.read_csv(processed / "products_skincare.csv", dtype={"product_id": "string"}, low_memory=False)
    split = temporal_split(interactions, args.split_quantile)
    print("Fitting TF-IDF on train-catalog metadata and positive train history...", flush=True)
    model = TfidfContentRecommender().fit(split.train, products)
    print("Evaluating personalized recommendations on the servable cohort...", flush=True)
    evaluation = evaluate_ranking(model, split.train, split.test, args.k, args.min_train_positives)
    results = {
        "model": "tfidf_content", "k": args.k, "split_quantile": args.split_quantile,
        "temporal_cutoff": split.cutoff.isoformat(), "min_train_positives": args.min_train_positives,
        "scikit_learn_version": sklearn.__version__,
        "vectorizer_configuration": VECTORIZER_CONFIG,
        "feature_fields": CONTENT_FIELDS,
        "vocabulary_size": len(model.vectorizer.vocabulary_),
        "user_profile_policy": "positive train only; rating 4 weight 1.0, rating 5 weight 2.0; weighted sum followed by L2 normalization",
        "unusable_profile_policy": "raise a clear error; no popularity fallback",
        "tfidf_fit_policy": "vocabulary and IDF fitted only on train-catalog product metadata",
        "candidate_policy": "all train products; no inventory filtering",
        "relevance_policy": "positive test products in train catalog; primary metrics, coverage, and latency use servable history-eligible users only",
        "diagnostic_policy": "cold-start user rate uses all history-eligible users; outside-catalog interaction rate uses all their positive test interactions",
        "ap_denominator": "min(k, number of in-catalog relevant products)",
        "train_interactions": len(split.train), "test_interactions": len(split.test),
        "train_positive_interactions": int(split.train["positive"].eq(1).sum()),
        "test_positive_interactions": int(split.test["positive"].eq(1).sum()),
        "train_max_timestamp": split.train["submission_time"].max().isoformat(),
        "test_min_timestamp": split.test["submission_time"].min().isoformat(),
        **evaluation,
        "experiment_runtime_seconds": perf_counter() - start,
    }
    comparison = compare_baseline(results, PROJECT_ROOT / "artifacts" / "baseline_metrics.json")
    if comparison:
        results["comparison_to_model_0"] = comparison
    print("\nGlowGuide Model 1: Personalized TF-IDF")
    print(f"Temporal cutoff: {split.cutoff}")
    for key in (
        "train_interactions", "test_interactions", "candidate_train_products",
        "history_eligible_users", "servable_evaluation_users", "cold_start_only_users",
        "eligible_positive_test_interactions", "relevant_test_interactions_used",
        "positive_test_interactions_outside_train_catalog",
    ):
        print(f"{key.replace('_', ' ').capitalize()}: {results[key]:,}")
    print(f"Cold-start-only user rate: {results['cold_start_only_user_rate']:.6%}")
    print(f"Outside-train-catalog positive interaction rate: {results['positive_test_interactions_outside_train_catalog_rate']:.6%}")
    print("Primary metrics: same servable cohort and train candidate catalog as Model 0.")
    for key, label in METRICS.items():
        suffix = f" | Model 0: {comparison[key]['model_0']:.12f}" if comparison else ""
        print(f"{label}@{args.k}: {results[key]:.12f}{suffix}")
    print(f"Mean recommendation latency: {results['mean_recommendation_latency_ms']:.6f} ms")
    print(f"P95 recommendation latency: {results['p95_recommendation_latency_ms']:.6f} ms")
    print(f"Experiment runtime (load through evaluation): {results['experiment_runtime_seconds']:.3f} s")
    output = PROJECT_ROOT / "artifacts" / "content_metrics.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("Saved artifacts/content_metrics.json")


if __name__ == "__main__":
    main()
