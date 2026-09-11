"""Load the catalog and review fields needed by preprocessing."""

from pathlib import Path

import pandas as pd


PRODUCT_COLUMNS = [
    "product_id", "product_name", "brand_name", "loves_count", "rating",
    "reviews", "ingredients", "price_usd", "highlights", "primary_category",
    "secondary_category", "tertiary_category", "out_of_stock",
]
REVIEW_COLUMNS = [
    "author_id", "product_id", "rating", "is_recommended", "submission_time",
    "skin_type", "skin_tone", "eye_color", "hair_color",
]
PRODUCT_REQUIRED_COLUMNS = ["product_id", "primary_category"]
REVIEW_REQUIRED_COLUMNS = ["author_id", "product_id", "rating", "submission_time"]


def _read_csv(path: Path, columns: list[str], required: list[str]) -> pd.DataFrame:
    """Select useful fields and preserve identifiers before type inference."""
    if not path.is_file():
        raise FileNotFoundError(f"Required CSV file not found: {path}")
    frame = pd.read_csv(
        path,
        usecols=lambda column: column in columns,
        dtype={"author_id": "string", "product_id": "string"},
        low_memory=False,
    )
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {', '.join(missing)}")
    return frame


def load_products(raw_dir: Path) -> pd.DataFrame:
    """Read product_info.csv without unused columns or exported indexes."""
    return _read_csv(raw_dir / "product_info.csv", PRODUCT_COLUMNS, PRODUCT_REQUIRED_COLUMNS)


def load_reviews(raw_dir: Path) -> pd.DataFrame:
    """Combine all reviews_*.csv files in deterministic filename order."""
    paths = sorted(raw_dir.glob("reviews_*.csv"))
    if not paths:
        raise FileNotFoundError(f"No reviews_*.csv files found in {raw_dir}")
    frames = [_read_csv(path, REVIEW_COLUMNS, REVIEW_REQUIRED_COLUMNS) for path in paths]
    return pd.concat(frames, ignore_index=True)
