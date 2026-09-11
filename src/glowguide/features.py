"""Deterministic product text without interaction or popularity statistics."""

import pandas as pd


CONTENT_FIELDS = [
    "product_name", "brand_name", "secondary_category", "tertiary_category",
    "ingredients", "highlights",
]


def build_product_features(products: pd.DataFrame) -> pd.DataFrame:
    """Return sorted product IDs and lowercase, whitespace-normalized text.

    Absent fields are empty. Ingredient punctuation is left for the vectorizer,
    preserving useful ingredient names without a hand-written removal list.
    """
    if "product_id" not in products:
        raise ValueError("Product metadata must contain product_id")
    ids = products["product_id"].astype("string").str.strip()
    if ids.isna().any() or ids.eq("").any() or ids.duplicated().any():
        raise ValueError("Product metadata requires unique, nonempty product IDs")
    fields = products.reindex(columns=CONTENT_FIELDS).astype("string").fillna("")
    text = fields.agg(" ".join, axis=1).astype("string")
    return pd.DataFrame({
        "product_id": ids,
        "feature_text": text.str.lower().str.replace(r"\s+", " ", regex=True).str.strip(),
    }).sort_values("product_id", kind="stable").reset_index(drop=True)
