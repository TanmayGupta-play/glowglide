"""Prepare skincare products and explainable interaction labels."""

import re

import pandas as pd

from .data import (
    PRODUCT_COLUMNS,
    PRODUCT_REQUIRED_COLUMNS,
    REVIEW_COLUMNS,
    REVIEW_REQUIRED_COLUMNS,
)


def normalize_profile_value(value: object) -> object:
    """Normalize a profile value to snake_case; preserve absent values as NA."""
    if pd.isna(value):
        return pd.NA
    text = str(value).strip()
    text = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", text)
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text)
    text = re.sub(r"[\s_-]+", "_", text).strip("_").lower()
    return text if text else pd.NA


def _require_columns(frame: pd.DataFrame, required: list[str], name: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {', '.join(missing)}")


def _clean_id(values: pd.Series) -> pd.Series:
    """Use nullable strings and treat blank identifiers as absent."""
    return values.astype("string").str.strip().replace("", pd.NA)


def clean_products(products: pd.DataFrame) -> pd.DataFrame:
    """Keep one row per skincare ID, retaining optional product metadata.

    Duplicate product IDs keep their first source row. Missing optional fields
    are retained as NA; they never disqualify a product.
    """
    _require_columns(products, PRODUCT_REQUIRED_COLUMNS, "Products")
    cleaned = products.reindex(columns=PRODUCT_COLUMNS).copy()
    cleaned["product_id"] = _clean_id(cleaned["product_id"])
    cleaned = cleaned.dropna(subset=["product_id"]).drop_duplicates("product_id")
    skincare = cleaned["primary_category"].astype("string").str.strip().str.casefold().eq("skincare")
    return cleaned.loc[skincare].reset_index(drop=True)


def clean_reviews(reviews: pd.DataFrame) -> pd.DataFrame:
    """Clean review fields and label preferences without requiring demographics.

    Missing/blank IDs, nonnumeric or nonfinite ratings, ratings outside 1-5,
    and invalid dates are removed. Optional fields remain nullable.
    """
    _require_columns(reviews, REVIEW_REQUIRED_COLUMNS, "Reviews")
    cleaned = reviews.reindex(columns=REVIEW_COLUMNS).copy()
    for column in ("author_id", "product_id"):
        cleaned[column] = _clean_id(cleaned[column])
    cleaned["rating"] = pd.to_numeric(cleaned["rating"], errors="coerce").replace(
        [float("inf"), float("-inf")], float("nan")
    )
    cleaned.loc[~cleaned["rating"].between(1, 5), "rating"] = pd.NA
    cleaned["is_recommended"] = pd.to_numeric(
        cleaned["is_recommended"], errors="coerce"
    ).astype("Float64")
    cleaned["submission_time"] = pd.to_datetime(cleaned["submission_time"], errors="coerce")
    cleaned = cleaned.dropna(subset=REVIEW_REQUIRED_COLUMNS).copy()
    for column in ("skin_type", "skin_tone", "eye_color", "hair_color"):
        # Profiles have few distinct values; normalize each value only once.
        values = cleaned[column].astype("string")
        mapping = {value: normalize_profile_value(value) for value in values.dropna().unique()}
        cleaned[column] = values.map(mapping).astype("string")
    cleaned["positive"] = cleaned["rating"].ge(4).astype("int8")
    # Missing recommendation values imply no strong label, not a negative rating.
    cleaned["strong_positive"] = (
        cleaned["rating"].eq(5) & cleaned["is_recommended"].eq(1)
    ).fillna(False).astype("int8")
    return cleaned.reset_index(drop=True)


def deduplicate_interactions(reviews: pd.DataFrame) -> pd.DataFrame:
    """Keep the latest parsed timestamp for each (author_id, product_id).

    Call after clean_reviews. Stable sorting makes timestamp ties deterministic:
    the last row in source order wins, including across sorted review files.
    """
    return (
        reviews.sort_values("submission_time", kind="stable")
        .drop_duplicates(["author_id", "product_id"], keep="last")
        .reset_index(drop=True)
    )


def filter_skincare_interactions(
    products: pd.DataFrame, reviews: pd.DataFrame
) -> pd.DataFrame:
    """Restrict interactions to IDs in a catalog produced by clean_products."""
    return reviews.loc[reviews["product_id"].isin(products["product_id"])].reset_index(drop=True)


def build_interactions(products: pd.DataFrame, reviews: pd.DataFrame) -> pd.DataFrame:
    """Clean raw frames, retain latest interactions, and enforce skincare scope."""
    products = clean_products(products)
    reviews = deduplicate_interactions(clean_reviews(reviews))
    return filter_skincare_interactions(products, reviews)
