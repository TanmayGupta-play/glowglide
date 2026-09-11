"""Interpretable fusion of fitted TRAIN-only recommendation components."""

from collections.abc import Collection, Mapping
from types import MappingProxyType

import numpy as np

from ..metrics import validate_k


COMPONENTS = ("popularity", "content", "profile", "collaborative")
NORMALIZATION_POLICY = "per user and component: raw / max(raw) if max > 0, otherwise zeros; before seen filtering"
UNAVAILABLE_POLICY = "renormalize configured weights over components with positive normalized signal; no available positive-weight component returns []"


def validate_weights(weights: Mapping[str, float]) -> np.ndarray:
    if set(weights) != set(COMPONENTS):
        raise ValueError(f"Weights must contain exactly {COMPONENTS}")
    try:
        values = np.array([weights[name] for name in COMPONENTS], dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Weights must be finite nonnegative numbers summing to 1") from exc
    if values.shape != (4,) or not np.isfinite(values).all() or (values < 0).any() or not np.isclose(values.sum(), 1, rtol=0, atol=1e-12):
        raise ValueError("Weights must be finite nonnegative numbers summing to 1")
    return values


def normalize_scores(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Normalize a component-by-product matrix without fitting any statistics."""
    raw = np.asarray(raw, dtype=float)
    if raw.ndim != 2 or raw.shape[0] != 4 or raw.shape[1] == 0 or not np.isfinite(raw).all():
        raise ValueError("Component scores must be finite aligned candidate vectors")
    maxima = raw.max(axis=1)
    available = maxima > 0
    normalized = np.zeros_like(raw)
    np.divide(raw, maxima[:, None], out=normalized, where=available[:, None])
    return normalized, available


def fuse_scores(normalized: np.ndarray, available: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Shared by live ranking and cached tuning, in fixed component order."""
    effective = weights * available
    total = effective.sum()
    if total > 0:
        effective = effective / total
    # Explicit fixed-order addition avoids BLAS thread-dependent reductions.
    scores = sum(effective[i] * normalized[i] for i in range(4))
    if not np.isfinite(scores).all():
        raise ValueError("Hybrid scores must be finite")
    return scores, effective


def rank_scores(scores: np.ndarray, product_ids: tuple[str, ...], seen_product_ids: Collection[str], k: int) -> list[str]:
    """Exact top K with stable ID ties; partition avoids sorting the full catalog."""
    validate_k(k)
    index = {p: i for i, p in enumerate(product_ids)}
    return rank_unseen_scores(scores, product_ids, [index[p] for p in seen_product_ids if p in index], k)


def rank_unseen_scores(scores: np.ndarray, product_ids: tuple[str, ...], seen_indices: Collection[int], k: int) -> list[str]:
    """Same ranking with pre-indexed seen items for repeated validation scoring."""
    mask = scores > 0
    mask[list(seen_indices)] = False
    eligible = np.flatnonzero(mask)
    if len(eligible) > k:
        threshold = np.partition(scores[eligible], len(eligible) - k)[len(eligible) - k]
        # Keep all cutoff ties before the final deterministic sort.
        eligible = eligible[scores[eligible] >= threshold]
    order = sorted(eligible, key=lambda i: (-scores[i], product_ids[i]))[:k]
    return [product_ids[i] for i in order]


class HybridRecommender:
    """Fuse four already fitted models; configured weights are immutable."""

    def __init__(self, components: Mapping, weights: Mapping[str, float]) -> None:
        if set(components) != set(COMPONENTS):
            raise ValueError(f"Components must contain exactly {COMPONENTS}")
        self._weights = validate_weights(weights)
        self.configured_weights = MappingProxyType(dict(zip(COMPONENTS, self._weights.tolist())))
        self.components = MappingProxyType(dict(components))
        first = components[COMPONENTS[0]]
        self.product_ids = tuple(first.product_ids)
        self.candidate_product_ids = frozenset(first.candidate_product_ids)
        self._product_index = {p: i for i, p in enumerate(self.product_ids)}
        if not self.product_ids or len(set(self.product_ids)) != len(self.product_ids):
            raise ValueError("Component product ordering must be nonempty and unique")
        for model in components.values():
            if frozenset(model.candidate_product_ids) != self.candidate_product_ids or set(model.product_ids) != self.candidate_product_ids:
                raise ValueError("Component candidate catalogs must match")
            if tuple(model.product_ids) != self.product_ids:
                raise ValueError("Component product ordering must match")

    def component_scores(self, user_id: str | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        raw = np.stack([self.components[name].score_candidates(user_id=user_id) for name in COMPONENTS])
        if raw.shape != (4, len(self.product_ids)):
            raise ValueError("Component scores must align with product_ids")
        normalized, available = normalize_scores(raw)
        return raw, normalized, available

    def score_candidates(self, *, user_id: str | None = None) -> np.ndarray:
        _, normalized, available = self.component_scores(user_id)
        return fuse_scores(normalized, available, self._weights)[0]

    def recommend(self, seen_product_ids: Collection[str], k: int = 10, *, user_id: str | None = None) -> list[str]:
        validate_k(k)
        seen = [self._product_index[p] for p in seen_product_ids if p in self._product_index]
        return rank_unseen_scores(self.score_candidates(user_id=user_id), self.product_ids, seen, k)

    def explain_score(self, user_id: str | None, product_id: str) -> dict:
        if product_id not in self.candidate_product_ids:
            raise ValueError("Explanation product must belong to the train catalog")
        index = self.product_ids.index(product_id)
        raw, normalized, available = self.component_scores(user_id)
        scores, effective = fuse_scores(normalized, available, self._weights)
        details = {}
        for i, name in enumerate(COMPONENTS):
            model = self.components[name]
            detail = {"raw_score": float(raw[i, index]), "normalized_score": float(normalized[i, index]),
                      "weighted_contribution": float(effective[i] * normalized[i, index]), "available": bool(available[i])}
            if name == "content" and available[i] and hasattr(model, "top_matching_terms"):
                detail["matching_terms"] = model.top_matching_terms(user_id, product_id)
            elif name in ("profile", "collaborative") and hasattr(model, "explain_score"):
                detail["explanation"] = model.explain_score(user_id, product_id)
            details[name] = detail
        return {"product_id": product_id, "configured_weights": dict(self.configured_weights),
                "effective_weights": dict(zip(COMPONENTS, effective.tolist())), "components": details,
                "final_score": float(scores[index])}
