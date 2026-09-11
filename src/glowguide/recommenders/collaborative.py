"""Explainable item-item collaborative filtering fitted exclusively on TRAIN."""

from collections.abc import Collection
from numbers import Real

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from ..metrics import validate_k


class ItemItemCollaborativeRecommender:
    """Binary positive-user cosine with fixed significance shrinkage (default 10).

    All user-level matrices are sparse. Similarity uses unique positive users,
    never negative rows. Processed input is normally unique per user/product;
    duplicate positive pairs defensively use their maximum rating for weights.
    Unknown/no-positive users return no recommendations, with no fallback.
    """

    def __init__(self, shrinkage: float = 10.0) -> None:
        if isinstance(shrinkage, bool) or not isinstance(shrinkage, Real) or not np.isfinite(shrinkage) or shrinkage < 0:
            raise ValueError("shrinkage must be a finite nonnegative number")
        self.shrinkage = float(shrinkage)
        self.candidate_product_ids: frozenset[str] = frozenset()
        self._fitted = False

    def fit(self, train: pd.DataFrame) -> "ItemItemCollaborativeRecommender":
        """Fit B and S from TRAIN only: Sij = co/sqrt(si*sj) * co/(co+shrinkage).

        The diagonal is zero; zero-support items remain in the catalog with
        zero similarities. No TEST frame or product metadata is accepted.
        """
        self._fitted = False
        required = ["author_id", "product_id", "positive", "rating"]
        if not set(required) <= set(train.columns):
            raise ValueError(f"Train data must contain {', '.join(required)}")
        if train.empty or train[required].isna().any().any():
            raise ValueError("Train data must be nonempty with valid IDs, labels, and ratings")
        frame = train[required].copy()
        if not frame["positive"].isin([0, 1]).all():
            raise ValueError("positive labels must be 0 or 1")
        for column in ("author_id", "product_id"):
            frame[column] = frame[column].astype("string")
        self.product_ids = tuple(sorted(frame["product_id"].unique()))
        self.candidate_product_ids = frozenset(self.product_ids)
        self._product_index = {p: i for i, p in enumerate(self.product_ids)}
        self._user_index = {u: i for i, u in enumerate(sorted(frame["author_id"].unique()))}
        positives = frame.loc[frame["positive"].eq(1)]
        if not positives["rating"].isin([4, 5]).all():
            raise ValueError("Positive training ratings must be 4 or 5")
        positives = positives.groupby(["author_id", "product_id"], as_index=False, sort=True)["rating"].max()
        rows = positives["author_id"].map(self._user_index).to_numpy(dtype=int)
        cols = positives["product_id"].map(self._product_index).to_numpy(dtype=int)
        shape = (len(self._user_index), len(self.product_ids))
        self.positive_matrix = csr_matrix((np.ones(len(positives)), (rows, cols)), shape=shape)
        self._positive_weights = csr_matrix((positives["rating"].map({4: 1.0, 5: 2.0}).to_numpy(dtype=float), (rows, cols)), shape=shape)
        self.support = np.asarray(self.positive_matrix.sum(axis=0)).ravel()
        co = (self.positive_matrix.T @ self.positive_matrix).tocoo()
        keep = co.row != co.col
        r, c, counts = co.row[keep], co.col[keep], co.data[keep]
        values = counts / np.sqrt(self.support[r] * self.support[c]) * (counts / (counts + self.shrinkage))
        self.item_similarity = csr_matrix((values, (r, c)), shape=(len(self.product_ids), len(self.product_ids)))
        self.item_similarity.sort_indices()
        self._fitted = True
        return self

    def user_profile(self, user_id: str | None) -> csr_matrix:
        """Return a copy of rating weights; unknown users have an empty profile."""
        if not self._fitted:
            raise RuntimeError("Fit the collaborative recommender first")
        if user_id not in self._user_index:
            return csr_matrix((1, len(self.product_ids)), dtype=float)
        return self._positive_weights.getrow(self._user_index[user_id])

    def recommend(self, seen_product_ids: Collection[str], k: int = 10, *, user_id: str | None = None) -> list[str]:
        """Rank positive weighted-average scores, then product ID ascending.

        All supplied seen products (positive and negative) are excluded. Only
        one catalog-sized score vector is dense; no zero-score filler is used.
        """
        validate_k(k)
        profile = self.user_profile(user_id)
        denominator = float(profile.sum())
        if denominator == 0:
            return []
        scores = (profile @ self.item_similarity).toarray().ravel() / denominator
        seen = set(seen_product_ids)
        order = np.argsort(-scores, kind="stable")
        return [self.product_ids[i] for i in order if scores[i] > 0 and self.product_ids[i] not in seen][:k]

    def explain_score(self, user_id: str | None, product_id: str, top_n: int = 5) -> dict:
        """Explain any train candidate, even if seen (recommend() excludes seen).

        weighted_contribution is weight * similarity, the numerator term.
        score_contribution divides that term by total profile weight. Omitted
        score mass is explicit so truncated explanations still reconstruct S.
        """
        validate_k(top_n)
        profile = self.user_profile(user_id)
        if product_id not in self._product_index:
            raise ValueError("Explanation product must belong to the train catalog")
        denominator = float(profile.sum())
        contributions = []
        for i, weight in zip(profile.indices, profile.data):
            similarity = float(self.item_similarity[i, self._product_index[product_id]])
            if similarity > 0:
                weighted = float(weight * similarity)
                contributions.append({"history_product_id": self.product_ids[i], "rating_weight": float(weight),
                                      "adjusted_similarity": similarity, "weighted_contribution": weighted,
                                      "score_contribution": weighted / denominator})
        contributions.sort(key=lambda x: (-x["weighted_contribution"], x["history_product_id"]))
        score = float((profile @ self.item_similarity[:, self._product_index[product_id]]).sum()) / denominator if denominator else 0.0
        shown = contributions[:top_n]
        return {"final_score": score, "profile_weight_sum": denominator, "contributions": shown,
                "other_score_contribution": score - sum(x["score_contribution"] for x in shown)}
