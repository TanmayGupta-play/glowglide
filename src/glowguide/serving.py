"""Deterministic Model 3-first routing, then business filters and explanations.

Raw strategy scores are not calibrated across strategies or probabilities.
All historical candidates are ranked (the catalog is small), ensuring filters
can search beyond the first K without altering scores or causing model switches.
"""

from copy import deepcopy
from numbers import Integral, Real

import numpy as np
import pandas as pd

from .preprocessing import normalize_profile_value
from .serving_artifacts import ServingBundle, optional_text, validate_serving_bundle


SCORE_TYPES = {"collaborative": "adjusted_collaborative_affinity", "content_fallback": "tfidf_cosine_similarity",
               "skin_profile": "smoothed_skin_profile_affinity", "popularity": "positive_user_count"}


def normalized_profile(value: str | None) -> str | None:
    """Use the project's one normalization rule, adapting missing NA to None."""
    value = normalize_profile_value(value)
    return None if pd.isna(value) else str(value)


class RecommendationService:
    def __init__(self, bundle: ServingBundle):
        validate_serving_bundle(bundle)
        self.bundle = bundle
        self.product_ids = tuple(bundle.popularity.product_ids)
        self._index = {p: i for i, p in enumerate(self.product_ids)}

    def health(self) -> dict:
        return {"status": "ok", "bundle_loaded": True, "bundle_version": self.bundle.build_metadata["bundle_version"],
                "candidate_products": len(self.product_ids), "historical_users": len(self.bundle.seen_by_user)}

    def product(self, product_id: str) -> dict | None:
        metadata = self.bundle.product_metadata.get(product_id)
        return deepcopy(metadata) if metadata is not None else None

    def _rank(self, scores, seen, *, positive_only=True):
        scores = np.asarray(scores, dtype=float)
        if scores.shape != (len(self.product_ids),) or not np.isfinite(scores).all():
            raise ValueError("Serving model returned invalid or nonfinite scores")
        order = np.argsort(-scores, kind="stable")
        return [self.product_ids[i] for i in order if self.product_ids[i] not in seen and (not positive_only or scores[i] > 0)]

    def _explanation(self, strategy, user_id, product_id, skin_type, skin_tone, score):
        if strategy == "collaborative":
            original = self.bundle.collaborative.explain_score(user_id, product_id, top_n=3)
            history = []
            for item in original["contributions"]:
                metadata = self.bundle.product_metadata.get(item["history_product_id"], {})
                history.append({"product_id": item["history_product_id"], "product_name": metadata.get("product_name"),
                                "rating_weight": item["rating_weight"], "similarity": item["adjusted_similarity"],
                                "weighted_contribution": item["weighted_contribution"], "score_contribution": item["score_contribution"]})
            return {"type": strategy, "because_you_liked": history, "final_score": original["final_score"],
                    "profile_weight_sum": original["profile_weight_sum"], "other_score_contribution": original["other_score_contribution"]}
        if strategy == "content_fallback":
            return {"type": strategy, "matching_terms": [term for term, _ in self.bundle.content.top_matching_terms(user_id, product_id, k=5)]}
        if strategy == "skin_profile":
            return {"type": strategy, **self.bundle.profile.explain_explicit_profile(product_id, skin_type, skin_tone)}
        return {"type": strategy, "positive_user_count": int(score)}

    def recommend(self, user_id: str | None = None, skin_type: str | None = None, skin_tone: str | None = None,
                  max_price: float | None = None, category: str | None = None, in_stock_only: bool = True, top_k: int = 10) -> dict:
        if isinstance(top_k, bool) or not isinstance(top_k, Integral) or not 1 <= top_k <= 50:
            raise ValueError("top_k must be an integer between 1 and 50")
        if max_price is not None and (isinstance(max_price, bool) or not isinstance(max_price, Real) or not np.isfinite(max_price) or max_price < 0):
            raise ValueError("max_price must be finite and nonnegative")
        user_id = optional_text(user_id)
        skin_type, skin_tone = normalized_profile(skin_type), normalized_profile(skin_tone)
        category = optional_text(category)
        category = category.casefold() if category else None
        seen = self.bundle.seen_by_user.get(user_id, frozenset())
        history_available = user_id in self.bundle.positive_history_users
        strategy, pool = None, []
        if history_available:
            scores = self.bundle.collaborative.score_candidates(user_id=user_id)
            pool = self._rank(scores, seen)
            if pool:
                strategy = "collaborative"
            else:
                scores = self.bundle.content.score_candidates(user_id=user_id)
                pool = self._rank(scores, seen)
                if np.any(scores > 0):
                    strategy = "content_fallback"
        if strategy is None:
            if skin_type is not None or skin_tone is not None:
                strategy = "skin_profile"
                scores = self.bundle.profile.score_explicit_profile(skin_type, skin_tone)
            else:
                strategy = "popularity"
                scores = self.bundle.popularity.score_candidates(user_id=user_id)
            pool = self._rank(scores, seen)
        # Routing is complete. Filters never change scores or choose a fallback.
        matches = []
        for product_id in pool:
            product = self.bundle.product_metadata[product_id]
            if in_stock_only and product["out_of_stock"] is True:
                continue
            if max_price is not None and (product["price_usd"] is None or product["price_usd"] > max_price):
                continue
            if category is not None and not any(
                optional_text(product.get(field)) and optional_text(product[field]).casefold() == category
                for field in ("secondary_category", "tertiary_category", "primary_category")
            ):
                continue
            score = float(scores[self._index[product_id]])
            matches.append({**deepcopy(product), "score": score, "score_type": SCORE_TYPES[strategy],
                            "explanation": self._explanation(strategy, user_id, product_id, skin_type, skin_tone, score)})
            if len(matches) == top_k:
                break
        return {"strategy": strategy, "requested_top_k": int(top_k), "returned_count": len(matches),
                "user_history_available": history_available,
                "skin_profile_used": {"skin_type": skin_type, "skin_tone": skin_tone} if strategy == "skin_profile" else None,
                "filters_applied": {"max_price": max_price, "price_field": "price_usd", "category": category, "in_stock_only": in_stock_only},
                "candidate_pool_size": len(pool), "bundle_version": self.bundle.build_metadata["bundle_version"],
                "recommendations": matches}
