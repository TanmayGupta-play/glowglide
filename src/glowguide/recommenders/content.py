"""Personalized TF-IDF cosine ranking from positive training history."""

from collections.abc import Collection

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from ..features import build_product_features
from ..metrics import validate_k


VECTORIZER_CONFIG = {
    "lowercase": True,
    "stop_words": "english",
    "ngram_range": (1, 2),
    "min_df": 1,
    "sublinear_tf": True,
    "norm": "l2",
}


class TfidfContentRecommender:
    """Fit train-catalog metadata and retain sparse positive user-item weights.

    Profiles are built on demand to avoid storing a large user-feature matrix.
    No popularity fallback is used: unknown or zero-vector profiles raise a
    clear error so this experiment measures content alone.
    """

    def __init__(self) -> None:
        self.candidate_product_ids: frozenset[str] = frozenset()
        self._fitted = False

    def fit(self, train: pd.DataFrame, products: pd.DataFrame) -> "TfidfContentRecommender":
        """Use only train products for vocabulary/IDF and positive train profiles."""
        self._fitted = False
        required = ["author_id", "product_id", "positive", "rating"]
        if not set(required) <= set(train.columns):
            raise ValueError(f"Train data must contain {', '.join(required)}")
        if train.empty or train[required].isna().any().any():
            raise ValueError("Train data must be nonempty with valid IDs, labels, and ratings")
        frame = train[required].copy()
        for column in ("author_id", "product_id"):
            frame[column] = frame[column].astype("string")
        self.product_ids = tuple(sorted(frame["product_id"].unique()))
        self.candidate_product_ids = frozenset(self.product_ids)
        self._product_index = {product: index for index, product in enumerate(self.product_ids)}
        features = build_product_features(products).set_index("product_id")
        # Missing metadata becomes an empty document, never a future candidate.
        texts = features.reindex(self.product_ids)["feature_text"].fillna("")
        self.vectorizer = TfidfVectorizer(**VECTORIZER_CONFIG)
        self.product_vectors = self.vectorizer.fit_transform(texts).tocsr()
        self._product_transpose = self.product_vectors.T.tocsr()
        self._terms = self.vectorizer.get_feature_names_out()
        positives = frame.loc[frame["positive"].eq(1)].sort_values(["author_id", "product_id", "rating"])
        if not positives["rating"].isin([4, 5]).all():
            raise ValueError("Positive training ratings must be 4 or 5 for the profile weighting policy")
        users = sorted(positives["author_id"].unique())
        self._user_index = {user: index for index, user in enumerate(users)}
        self._positive_weights = csr_matrix((
            positives["rating"].map({4: 1.0, 5: 2.0}).to_numpy(),
            (positives["author_id"].map(self._user_index).to_numpy(dtype=int),
             positives["product_id"].map(self._product_index).to_numpy(dtype=int)),
        ), shape=(len(users), len(self.product_ids)))
        self._fitted = True
        return self

    def user_profile(self, user_id: str | None) -> csr_matrix:
        """L2-normalize the sum of rating-weighted positive train product vectors."""
        if not self._fitted:
            raise RuntimeError("Fit the content recommender before requesting a user profile")
        if user_id not in self._user_index:
            raise ValueError(f"No positive training profile for user {user_id!r}")
        profile = self._positive_weights.getrow(self._user_index[user_id]) @ self.product_vectors
        profile.eliminate_zeros()
        if profile.nnz == 0:
            raise ValueError(f"No usable metadata profile for user {user_id!r}")
        return normalize(profile, norm="l2", copy=False)

    def recommend(self, seen_product_ids: Collection[str], k: int = 10, *, user_id: str | None = None) -> list[str]:
        """Score one user's candidates, excluding all supplied train history.

        Only a single catalog-sized score array is dense. Candidate rows are
        sorted by ID, so stable score sorting also resolves ties by ID.
        """
        validate_k(k)
        profile = self.user_profile(user_id)
        scores = (profile @ self._product_transpose).toarray().ravel()
        seen = set(seen_product_ids)
        order = np.argsort(-scores, kind="stable")
        return [self.product_ids[index] for index in order if self.product_ids[index] not in seen][:k]

    def top_matching_terms(self, user_id: str, product_id: str, k: int = 5) -> list[tuple[str, float]]:
        """Return shared terms ordered by their contribution to cosine similarity."""
        validate_k(k)
        profile = self.user_profile(user_id)
        if product_id not in self._product_index:
            raise ValueError("Explanation product must belong to the train candidate catalog")
        contributions = profile.multiply(self.product_vectors.getrow(self._product_index[product_id])).tocsr()
        terms = [
            (str(self._terms[index]), float(value))
            for index, value in zip(contributions.indices, contributions.data) if value > 0
        ]
        return sorted(terms, key=lambda pair: (-pair[1], pair[0]))[:k]
