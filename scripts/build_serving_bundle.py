#!/usr/bin/env python3
"""Build trusted local serving artifacts after offline architecture selection."""

import json
from pathlib import Path
import sys
from time import perf_counter

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from glowguide.serving_artifacts import build_serving_bundle, save_serving_bundle


def main():
    start = perf_counter()
    processed = PROJECT_ROOT / "data" / "processed"
    interactions = pd.read_csv(processed / "interactions.csv", dtype={name: "string" for name in (
        "author_id", "product_id", "skin_type", "skin_tone", "eye_color", "hair_color")}, low_memory=False)
    products = pd.read_csv(processed / "products_skincare.csv", dtype={"product_id": "string"}, low_memory=False)
    print("Fitting Models 0-3 on ALL processed historical interactions; no temporal holdout for serving.", flush=True)
    bundle = build_serving_bundle(interactions, products)
    output = PROJECT_ROOT / "artifacts" / "serving_bundle.joblib"
    print("Saving trusted local serving bundle...", flush=True)
    save_serving_bundle(bundle, output)
    manifest = {**bundle.build_metadata, "bundle_file_size_bytes": output.stat().st_size,
                "build_runtime_seconds": perf_counter() - start}
    output.with_name("serving_manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for key in ("interaction_rows", "historical_users", "candidate_products", "full_skincare_catalog_products",
                "catalog_products_without_historical_interactions", "bundle_file_size_bytes", "build_runtime_seconds"):
        print(f"{key}: {manifest[key]}", flush=True)
    print("Saved artifacts/serving_bundle.joblib and artifacts/serving_manifest.json", flush=True)


if __name__ == "__main__":
    main()
