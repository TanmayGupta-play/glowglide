#!/usr/bin/env python3
"""Build processed CSVs with `python scripts/build_processed_data.py`."""

from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Support this repository's src layout without requiring a package installation.
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from glowguide.data import load_products, load_reviews
from glowguide.preprocessing import (
    clean_products,
    clean_reviews,
    deduplicate_interactions,
    filter_skincare_interactions,
)


def print_summary(counts: dict[str, int], interactions: pd.DataFrame) -> None:
    """Report row accounting and profile coverage on the final interactions."""
    print("\nGlowGuide preprocessing summary")
    print("-" * 52)
    for label, count in counts.items():
        print(f"{label:<34} {count:>15,}")
    for label, count in {
        "Unique users": interactions["author_id"].nunique(),
        "Unique products": interactions["product_id"].nunique(),
        "Positive interactions": interactions["positive"].sum(),
        "Strong positive interactions": interactions["strong_positive"].sum(),
    }.items():
        print(f"{label:<34} {count:>15,}")
    dates = interactions["submission_time"]
    date_range = "N/A (no interactions)" if dates.empty else f"{dates.min()} to {dates.max()}"
    print(f"Date range: {date_range}")
    for column in ("skin_type", "skin_tone"):
        coverage = interactions[column].notna().mean() if len(interactions) else 0.0
        print(f"{column.replace('_', ' ').capitalize()} coverage: {coverage:.2%}")


def main() -> None:
    raw_dir = PROJECT_ROOT / "data" / "raw"
    processed_dir = PROJECT_ROOT / "data" / "processed"
    print("Loading raw products and reviews...", flush=True)
    products = load_products(raw_dir)
    reviews = load_reviews(raw_dir)
    counts = {"Raw products": len(products), "Raw review rows": len(reviews)}

    print("Cleaning and deduplicating interactions...", flush=True)
    products = clean_products(products)
    counts["Skincare products"] = len(products)
    reviews = clean_reviews(reviews)
    counts["Invalid review rows removed"] = counts["Raw review rows"] - len(reviews)
    clean_count = len(reviews)
    reviews = deduplicate_interactions(reviews)
    counts["Duplicates removed"] = clean_count - len(reviews)
    interactions = filter_skincare_interactions(products, reviews)
    counts["Outside skincare catalog removed"] = len(reviews) - len(interactions)
    counts["Processed interactions"] = len(interactions)

    processed_dir.mkdir(parents=True, exist_ok=True)
    products.to_csv(processed_dir / "products_skincare.csv", index=False)
    interactions.to_csv(processed_dir / "interactions.csv", index=False)
    print_summary(counts, interactions)
    print(f"Products shape: {products.shape}")
    print(f"Interactions shape: {interactions.shape}")
    print("Wrote data/processed/products_skincare.csv and data/processed/interactions.csv")


if __name__ == "__main__":
    main()
