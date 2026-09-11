"""Train-only popularity based on distinct positive users."""

from collections.abc import Collection

import pandas as pd
import numpy as np

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
        self._product_ids = tuple(catalog)
        self._scores = np.array([counts.get(product, 0) for product in catalog], dtype=float)
        self._fitted = True
        return self

    @property
    def product_ids(self) -> tuple[str, ...]:
        """Read-only catalog alignment for score_candidates()."""
        return getattr(self, "_product_ids", ())

    def score_candidates(self, *, user_id: str | None = None) -> np.ndarray:
        """Distinct positive TRAIN-user counts, with no seen filtering."""
        if not self._fitted:
            raise RuntimeError("Fit the popularity recommender first")
        return self._scores.copy()

    def recommend(self, seen_product_ids: Collection[str], k: int = 10, *, user_id: str | None = None) -> list[str]:
        """Return unseen products; user_id is accepted but does not affect popularity."""
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
