"""Binary-relevance ranking metrics for a single user."""

from collections.abc import Collection, Sequence
from math import log2
from numbers import Integral


def validate_k(k: int) -> None:
    if isinstance(k, bool) or not isinstance(k, Integral) or k <= 0:
        raise ValueError("k must be a positive integer")


def _hits(recommended: Sequence[str], relevant: Collection[str], k: int) -> list[int]:
    """Count a repeated product only at its first rank; repeats waste a slot."""
    validate_k(k)
    relevant = set(relevant)
    seen = set()
    hits = []
    for product in recommended[:k]:
        hits.append(int(product in relevant and product not in seen))
        seen.add(product)
    return hits


def precision_at_k(recommended: Sequence[str], relevant: Collection[str], k: int) -> float:
    """Relevant hits divided by K, even if fewer than K items are returned."""
    return sum(_hits(recommended, relevant, k)) / k


def recall_at_k(recommended: Sequence[str], relevant: Collection[str], k: int) -> float:
    """Relevant hits divided by the number of distinct relevant products."""
    hits = _hits(recommended, relevant, k)
    return sum(hits) / len(set(relevant)) if relevant else 0.0


def hit_rate_at_k(recommended: Sequence[str], relevant: Collection[str], k: int) -> float:
    """One if at least one relevant product appears, otherwise zero."""
    return float(any(_hits(recommended, relevant, k)))


def ndcg_at_k(recommended: Sequence[str], relevant: Collection[str], k: int) -> float:
    """Binary DCG normalized by an ideal ranking of min(K, |relevant|) hits."""
    hits = _hits(recommended, relevant, k)
    ideal = sum(1 / log2(rank + 1) for rank in range(1, min(k, len(set(relevant))) + 1))
    dcg = sum(hit / log2(rank + 1) for rank, hit in enumerate(hits, start=1))
    return dcg / ideal if ideal else 0.0


def average_precision_at_k(recommended: Sequence[str], relevant: Collection[str], k: int) -> float:
    """Sum precision at relevant ranks / min(K, |relevant|); zero if empty.

    The evaluator averages this per-user value to obtain MAP@K.
    """
    hits = _hits(recommended, relevant, k)
    denominator = min(k, len(set(relevant)))
    running_hits = 0
    total = 0.0
    for rank, hit in enumerate(hits, start=1):
        running_hits += hit
        if hit:
            total += running_hits / rank
    return total / denominator if denominator else 0.0
