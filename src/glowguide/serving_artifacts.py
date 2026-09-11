"""Build and persist serving models from trusted local processed history.

SECURITY: joblib uses pickle-style deserialization, which can execute code.
Only load bundles built from trusted local project data; never accept uploaded
or otherwise untrusted artifact paths. Validation is not a pickle sandbox.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
import platform

import joblib
import numpy as np
import pandas as pd

from .recommenders.popularity import MostPopularRecommender
from .recommenders.content import TfidfContentRecommender, VECTORIZER_CONFIG
from .recommenders.profile import SkinProfileRecommender, SMOOTHING, SCORE_WEIGHTS
from .recommenders.collaborative import ItemItemCollaborativeRecommender


BUNDLE_VERSION = "1"
TRAINING_SCOPE = "all available processed historical interactions after offline model selection"
MODEL_CLASSES = {"popularity": MostPopularRecommender, "content": TfidfContentRecommender,
                 "profile": SkinProfileRecommender, "collaborative": ItemItemCollaborativeRecommender}
TEXT_FIELDS = ("product_name", "brand_name", "primary_category", "secondary_category", "tertiary_category")


def dependency_versions() -> dict:
    return {"python": platform.python_version(), **{name: version(name) for name in (
        "numpy", "pandas", "scipy", "scikit-learn", "joblib")}}


def optional_text(value) -> str | None:
    if value is None or pd.isna(value):
        return None
    return " ".join(str(value).split()) or None


def optional_number(value) -> float | None:
    try:
        number = float(value)
        return number if np.isfinite(number) else None
    except (ValueError, TypeError):
        return None


def stock_flag(value) -> bool | None:
    """Handle nullable numeric, bool and CSV string flags without bool('0')."""
    text = optional_text(value)
    if text is None:
        return None
    if text.casefold() in ("true", "yes"):
        return True
    if text.casefold() in ("false", "no"):
        return False
    number = optional_number(value)
    return bool(number) if number in (0, 1) else None


def metadata_index(products: pd.DataFrame, candidates) -> dict[str, dict]:
    """Index full skincare metadata plus metadata-empty historical candidates.

    price_usd is the actual processed field; no conversion or price inference.
    Missing stock status is unknown, not a claim that a product is in stock.
    """
    if "product_id" not in products or products.product_id.isna().any() or products.product_id.astype(str).duplicated().any():
        raise ValueError("Product metadata requires unique nonmissing product IDs")
    indexed = {}
    for row in products.to_dict(orient="records"):
        product_id = str(row["product_id"])
        indexed[product_id] = {"product_id": product_id, **{name: optional_text(row.get(name)) for name in TEXT_FIELDS},
                               "price_usd": optional_number(row.get("price_usd")),
                               "rating": optional_number(row.get("rating")), "out_of_stock": stock_flag(row.get("out_of_stock"))}
    for product_id in candidates:
        indexed.setdefault(product_id, {"product_id": product_id, **dict.fromkeys(TEXT_FIELDS),
                                        "price_usd": None, "rating": None, "out_of_stock": None})
    return indexed


@dataclass
class ServingBundle:
    popularity: MostPopularRecommender
    content: TfidfContentRecommender
    profile: SkinProfileRecommender
    collaborative: ItemItemCollaborativeRecommender
    product_metadata: dict[str, dict]
    seen_by_user: dict[str, frozenset[str]]
    positive_history_users: frozenset[str]
    build_metadata: dict


def validate_serving_bundle(bundle: ServingBundle) -> None:
    """Fail clearly on unsupported local serialization versions or alignment."""
    if not isinstance(bundle, ServingBundle) or bundle.build_metadata.get("bundle_version") != BUNDLE_VERSION:
        raise ValueError("Incompatible serving bundle version; rebuild the bundle")
    if bundle.build_metadata.get("dependency_versions") != dependency_versions():
        raise ValueError("Incompatible serving bundle dependency versions; rebuild in this environment")
    ordering = tuple(bundle.popularity.product_ids)
    catalog = frozenset(ordering)
    if not ordering or len(ordering) != len(catalog):
        raise ValueError("Serving bundle has an invalid candidate catalog")
    for name, cls in MODEL_CLASSES.items():
        model = getattr(bundle, name)
        if not isinstance(model, cls) or not model._fitted or tuple(model.product_ids) != ordering or model.candidate_product_ids != catalog:
            raise ValueError("Serving bundle models must be fitted with identical catalog ordering")
    if not catalog <= bundle.product_metadata.keys() or not bundle.positive_history_users <= bundle.seen_by_user.keys():
        raise ValueError("Serving bundle is missing product metadata or historical seen sets")


def build_serving_bundle(interactions: pd.DataFrame, products: pd.DataFrame) -> ServingBundle:
    """Fit Models 0-3 on ALL supplied processed history; no evaluation split."""
    if interactions.empty:
        raise ValueError("Cannot build a serving bundle from empty history")
    history = interactions.copy()
    for name in ("author_id", "product_id"):
        history[name] = history[name].astype("string")
    history["submission_time"] = pd.to_datetime(history["submission_time"], errors="raise", format="mixed")
    models = {"popularity": MostPopularRecommender().fit(history),
              "content": TfidfContentRecommender().fit(history, products),
              "profile": SkinProfileRecommender().fit(history),
              "collaborative": ItemItemCollaborativeRecommender(shrinkage=10.0).fit(history)}
    candidates = models["popularity"].candidate_product_ids
    full_catalog = frozenset(products.product_id.astype(str))
    seen = history.groupby("author_id").product_id.agg(frozenset).to_dict()
    metadata = {
        "bundle_version": BUNDLE_VERSION, "training_scope": TRAINING_SCOPE,
        "interaction_rows": len(history), "historical_users": len(seen), "candidate_products": len(candidates),
        "full_skincare_catalog_products": len(full_catalog),
        "catalog_products_without_historical_interactions": len(full_catalog - candidates),
        "positive_interactions": int(history.positive.eq(1).sum()),
        "min_timestamp": history.submission_time.min().isoformat(), "max_timestamp": history.submission_time.max().isoformat(),
        "primary_strategy": "item_item_collaborative",
        "fallback_order": ["content_fallback", "skin_profile_if_explicitly_supplied", "popularity"],
        "routing_policy": "collaborative for positive history with unseen positive signal; otherwise usable content; otherwise explicit request profile or popularity; no switch after business filters",
        "collaborative_shrinkage": 10.0, "content_vectorizer_configuration": VECTORIZER_CONFIG,
        "profile_smoothing_configuration": SMOOTHING, "profile_score_weights": SCORE_WEIGHTS,
        "model_class_names": {name: cls.__name__ for name, cls in MODEL_CLASSES.items()},
        "dependency_versions": dependency_versions(), "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "price_field": "price_usd", "price_policy": "USD as provided by processed dataset; no currency conversion",
        "stock_policy": "in_stock_only excludes explicitly marked out-of-stock products; missing status remains unknown",
        "candidate_policy": "all products with historical interactions; cold catalog metadata is queryable but not ranked",
    }
    bundle = ServingBundle(**models, product_metadata=metadata_index(products, candidates), seen_by_user=seen,
                           positive_history_users=frozenset(history.loc[history.positive.eq(1), "author_id"]), build_metadata=metadata)
    validate_serving_bundle(bundle)
    return bundle


def save_serving_bundle(bundle: ServingBundle, path: str | Path) -> None:
    """Persist a locally built trusted bundle; does not touch benchmark JSON."""
    validate_serving_bundle(bundle)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path, compress=3)


def load_serving_bundle(path: str | Path) -> ServingBundle:
    """Load ONLY trusted local artifacts. Never use this with user uploads."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Serving bundle missing: {path}. Run scripts/build_serving_bundle.py first.")
    try:
        bundle = joblib.load(path)
        validate_serving_bundle(bundle)
    except Exception as exc:
        raise RuntimeError(f"Cannot load serving bundle: corrupt or incompatible artifact. Rebuild it. {exc}") from exc
    return bundle
