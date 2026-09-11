"""Hierarchically smoothed skin-profile affinity from training outcomes."""

from collections.abc import Collection

import numpy as np
import pandas as pd

from ..metrics import validate_k
from ..profile import infer_user_profiles


SMOOTHING = {"overall": 20.0, "skin_type": 20.0, "skin_tone": 20.0, "exact": 40.0}
SCORE_WEIGHTS = {"overall": 0.10, "skin_type": 0.25, "skin_tone": 0.15, "exact": 0.50}


class SkinProfileRecommender:
    """Learn rates from both positive and negative train interactions.

    Every training row is attributed to its author's TRAIN-inferred modal
    profile. Single-attribute cells shrink toward product overall; exact cells
    shrink toward the mean of smoothed type and tone rates. Zero-count cells
    equal their prior. Fixed pseudo-counts regularize sparse cells without test
    tuning. Rankings for each inferred profile are precomputed during fitting.
    """

    def __init__(self) -> None:
        self.candidate_product_ids: frozenset[str] = frozenset()
        self._fitted = False

    @staticmethod
    def _profile_key(skin_type: object, skin_tone: object) -> tuple[str | None, str | None]:
        return (None if pd.isna(skin_type) else str(skin_type), None if pd.isna(skin_tone) else str(skin_tone))

    def fit(self, train: pd.DataFrame) -> "SkinProfileRecommender":
        """Fit only TRAIN data; never accepts test data or a future catalog."""
        self._fitted = False
        required = {"author_id", "product_id", "positive", "submission_time"}
        if not required <= set(train.columns) or train.empty:
            raise ValueError("Nonempty train data requires author_id, product_id, positive, submission_time")
        frame = train.copy()
        frame["author_id"] = frame["author_id"].astype("string")
        frame["product_id"] = frame["product_id"].astype("string")
        if frame["product_id"].isna().any() or frame["product_id"].eq("").any() or not frame["positive"].isin([0, 1]).all():
            raise ValueError("Training product IDs must be nonempty and positive labels must be binary")
        self.user_profiles = infer_user_profiles(frame)
        self._users = {
            user: self._profile_key(skin_type, skin_tone)
            for user, skin_type, skin_tone in self.user_profiles.itertuples(name=None)
        }
        for attribute in ("skin_type", "skin_tone"):
            frame[attribute] = frame["author_id"].map(self.user_profiles[attribute])
        self.product_ids = tuple(sorted(frame["product_id"].unique()))
        self.candidate_product_ids = frozenset(self.product_ids)
        self._product_index = {product: index for index, product in enumerate(self.product_ids)}
        self.global_positive_rate = float(frame["positive"].mean())
        self._stats = {}
        for signal, attributes in {
            "overall": [], "skin_type": ["skin_type"], "skin_tone": ["skin_tone"],
            "exact": ["skin_type", "skin_tone"],
        }.items():
            table = frame.groupby(attributes + ["product_id"])["positive"].agg(positive_count="sum", interaction_count="size")
            groups = [((), table)] if not attributes else table.groupby(level=list(range(len(attributes))))
            for key, group in groups:
                key = key if isinstance(key, tuple) else (key,)
                if attributes:
                    group = group.droplevel(list(range(len(attributes))))
                aligned = group.reindex(self.product_ids, fill_value=0)
                self._stats[(signal, key)] = (
                    aligned["positive_count"].to_numpy(dtype=float), aligned["interaction_count"].to_numpy(dtype=float),
                )
        self._zero = np.zeros(len(self.product_ids))
        self._rankings = {}
        for profile in set(self._users.values()) | {(None, None)}:
            components = self._components(profile)
            scores = self._combine(components)
            self._rankings[profile] = tuple(self.product_ids[i] for i in np.argsort(-scores, kind="stable"))
        self._fitted = True
        return self

    def _counts(self, signal: str, key: tuple) -> tuple[np.ndarray, np.ndarray]:
        return self._stats.get((signal, key), (self._zero, self._zero))

    def _rate(self, signal: str, key: tuple, prior: float | np.ndarray) -> np.ndarray:
        positive, count = self._counts(signal, key)
        strength = SMOOTHING[signal]
        smoothed = (positive + strength * prior) / (count + strength)
        # Empty cells back off exactly, without roundoff changing tied scores.
        return np.where(count == 0, prior, smoothed)

    def _components(self, profile: tuple) -> dict[str, np.ndarray]:
        skin_type, skin_tone = profile
        rates = {"overall": self._rate("overall", (), self.global_positive_rate)}
        if skin_type is not None:
            rates["skin_type"] = self._rate("skin_type", (skin_type,), rates["overall"])
        if skin_tone is not None:
            rates["skin_tone"] = self._rate("skin_tone", (skin_tone,), rates["overall"])
        if skin_type is not None and skin_tone is not None:
            prior = (rates["skin_type"] + rates["skin_tone"]) / 2
            rates["exact"] = self._rate("exact", profile, prior)
        return rates

    @staticmethod
    def _combine(components: dict[str, np.ndarray]) -> np.ndarray:
        denominator = sum(SCORE_WEIGHTS[name] for name in components)
        return sum(SCORE_WEIGHTS[name] * values for name, values in components.items()) / denominator

    def _user_profile(self, user_id: str | None) -> tuple:
        if not self._fitted:
            raise RuntimeError("Fit the skin-profile recommender before requesting recommendations")
        # Unknown users have no TRAIN-inferred attributes and use overall only.
        return self._users.get(user_id, (None, None))

    def recommend(self, seen_product_ids: Collection[str], k: int = 10, *, user_id: str | None = None) -> list[str]:
        """Return up to K unseen train products with product-ID tie-breaking."""
        validate_k(k)
        profile = self._user_profile(user_id)
        ranking = self._rankings[profile]
        seen = set(seen_product_ids)
        result = []
        for product in ranking:
            if product not in seen:
                result.append(product)
                if len(result) == k:
                    break
        return result

    def explain_score(self, user_id: str | None, product_id: str) -> dict:
        """Expose actual smoothed components, weights, counts, and final score."""
        profile = self._user_profile(user_id)
        if product_id not in self._product_index:
            raise ValueError("Explanation product must belong to the train catalog")
        index = self._product_index[product_id]
        rates = self._components(profile)
        keys = {"overall": (), "skin_type": (profile[0],), "skin_tone": (profile[1],), "exact": profile}
        total_weight = sum(SCORE_WEIGHTS[name] for name in rates)
        signals = {}
        for name in SCORE_WEIGHTS:
            if name not in rates:
                signals[name] = None
                continue
            positive, count = self._counts(name, keys[name])
            signals[name] = {
                "smoothed_rate": float(rates[name][index]),
                "positive_count": int(positive[index]), "interaction_count": int(count[index]),
                "weight": SCORE_WEIGHTS[name] / total_weight,
            }
        return {
            "skin_type": profile[0], "skin_tone": profile[1],
            "global_train_positive_rate": self.global_positive_rate,
            "signals": signals, "final_score": float(self._combine(rates)[index]),
        }
