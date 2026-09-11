"""Train-only popularity based on distinct positive users."""

from collections.abc import Collection

import pandas as pd

from ..metrics import validate_k


class MostPopularRecommender:
    """Rank every train product by positive-user count, then product ID.

    Products without positive training interactions receive zero counts and
    remain candidates. Inventory and test information are never used.
    """

    def __init__(self) -> None:
        self.ranked_product_ids: tuple[str, ...] = ()
        self.candidate_product_ids: frozenset[str] = frozenset()
        self._fitted = False

    def fit(self, train: pd.DataFrame) -> "MostPopularRecommender":
        required = ["author_id", "product_id", "positive"]
        missing = sorted(set(required) - set(train.columns))
        if missing:
            raise ValueError(f"Train data is missing required columns: {', '.join(missing)}")
        if train.empty:
            raise ValueError("Cannot fit popularity on empty train data")
        if train[required].isna().any().any():
            raise ValueError("Train author_id, product_id, and positive must not be missing")
        frame = train[required].copy()
        for column in ("author_id", "product_id"):
            frame[column] = frame[column].astype("string")
        counts = frame.loc[frame["positive"].eq(1)].groupby("product_id")["author_id"].nunique()
        catalog = sorted(frame["product_id"].unique())
        self.ranked_product_ids = tuple(sorted(catalog, key=lambda product: (-counts.get(product, 0), product)))
        self.candidate_product_ids = frozenset(catalog)
        self._fitted = True
        return self

    def recommend(self, seen_product_ids: Collection[str], k: int = 10) -> list[str]:
        """Return up to K distinct unseen products; seen includes negative history."""
        validate_k(k)
        if not self._fitted:
            raise RuntimeError("Fit the popularity recommender before requesting recommendations")
        seen = set(seen_product_ids)
        recommended = []
        for product in self.ranked_product_ids:
            if product not in seen:
                recommended.append(product)
                if len(recommended) == k:
                    break
        return recommended
