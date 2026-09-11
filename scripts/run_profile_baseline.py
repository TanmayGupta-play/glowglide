#!/usr/bin/env python3
"""Evaluate Model 2 under the frozen temporal and servable-cohort benchmark."""

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from glowguide.evaluation import eligible_evaluation_users, evaluate_ranking
from glowguide.metrics import validate_k
from glowguide.recommenders.profile import SkinProfileRecommender, SMOOTHING, SCORE_WEIGHTS
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
        "baseline": baseline[key], "model_2": results[key], "absolute_change": results[key] - baseline[key],
        "percentage_change": (results[key] / baseline[key] - 1) * 100 if baseline[key] else None,
    } for key in METRICS}


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
    print("Inferring TRAIN profiles and fitting smoothed affinity statistics...", flush=True)
    model = SkinProfileRecommender().fit(split.train)
    print("Evaluating the unchanged servable cohort...", flush=True)
    evaluation = evaluate_ranking(model, split.train, split.test, args.k, args.min_train_positives)
    history = eligible_evaluation_users(split.train, split.test, args.min_train_positives)
    reachable_users = split.test.loc[
        split.test["positive"].eq(1) & split.test["product_id"].isin(model.candidate_product_ids), "author_id"
    ]
    profiles = model.user_profiles.loc[history[history.isin(reachable_users)]]
    has_type, has_tone = profiles["skin_type"].notna(), profiles["skin_tone"].notna()
    coverage = {name: {"count": int(mask.sum()), "percentage": float(mask.mean() * 100)} for name, mask in {
        "both": has_type & has_tone, "type_only": has_type & ~has_tone,
        "tone_only": ~has_type & has_tone, "neither": ~has_type & ~has_tone,
    }.items()}
    results = {
        "model": "skin_profile_affinity", "k": args.k, "split_quantile": args.split_quantile,
        "temporal_cutoff": split.cutoff.isoformat(), "min_train_positives": args.min_train_positives,
        "smoothing_configuration": SMOOTHING, "score_weights": SCORE_WEIGHTS,
        "smoothing_policy": "(positive_count + strength * prior_rate) / (interaction_count + strength); overall to global; type/tone to overall; exact to mean(type,tone)",
        "profile_inference_policy": "independent non-null TRAIN modes; frequency ties use latest tied-value observation; timestamp ties use lexical value order",
        "affinity_group_policy": "all TRAIN interactions attributed to author TRAIN-inferred modal profile; positive and negative labels included",
        "missing_profile_policy": "renormalize available signal weights; neither or unknown user uses smoothed overall only",
        "candidate_policy": "all train products; no inventory filtering",
        "relevance_policy": "positive test products in train catalog; primary metrics, coverage, and latency use servable history-eligible users only",
        "ap_denominator": "min(k, number of in-catalog relevant products)",
        "train_interactions": len(split.train), "test_interactions": len(split.test),
        "train_positive_interactions": int(split.train["positive"].sum()),
        "test_positive_interactions": int(split.test["positive"].sum()),
        "train_max_timestamp": split.train["submission_time"].max().isoformat(),
        "test_min_timestamp": split.test["submission_time"].min().isoformat(),
        "profile_coverage": coverage, **evaluation,
        "experiment_runtime_seconds": perf_counter() - start,
    }
    results["comparisons"] = {}
    for name, filename in (("most_popular", "baseline_metrics.json"), ("tfidf_content", "content_metrics.json")):
        comparison = compare_artifact(results, PROJECT_ROOT / "artifacts" / filename, name)
        if comparison:
            results["comparisons"][name] = comparison
    print("\nGlowGuide Model 2: Skin Profile Affinity")
    print(f"Temporal cutoff: {split.cutoff}")
    for key in (
        "train_interactions", "test_interactions", "candidate_train_products", "history_eligible_users",
        "servable_evaluation_users", "cold_start_only_users", "eligible_positive_test_interactions",
        "relevant_test_interactions_used", "positive_test_interactions_outside_train_catalog",
    ):
        print(f"{key.replace('_', ' ').capitalize()}: {results[key]:,}")
    for key in ("cold_start_only_user_rate", "positive_test_interactions_outside_train_catalog_rate"):
        print(f"{key.replace('_', ' ').capitalize()}: {results[key]:.6%}")
    print("TRAIN profile coverage among servable users:")
    for name, values in coverage.items():
        print(f"  {name}: {values['count']:,} ({values['percentage']:.6f}%)")
    for key, label in METRICS.items():
        print(f"{label}@{args.k}: {results[key]:.15f}")
        for name, comparison in results["comparisons"].items():
            print(f"  versus {name}: {comparison[key]['absolute_change']:+.12f} absolute; {comparison[key]['percentage_change']:+.4f}%" if comparison[key]['percentage_change'] is not None else f"  versus {name}: baseline is zero")
    print(f"Mean recommendation latency: {results['mean_recommendation_latency_ms']:.6f} ms")
    print(f"P95 recommendation latency: {results['p95_recommendation_latency_ms']:.6f} ms")
    print(f"Experiment runtime (load through evaluation): {results['experiment_runtime_seconds']:.3f} s")
    output = PROJECT_ROOT / "artifacts" / "profile_metrics.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("Saved artifacts/profile_metrics.json")


if __name__ == "__main__":
    main()
