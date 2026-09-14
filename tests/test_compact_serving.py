"""Serving-only storage must preserve fitted algorithms and serialization."""

from unittest.mock import patch

import joblib
import numpy as np
import pytest

from glowguide.compact_serving import compact_serving_bundle, unsigned_dtype
from glowguide.serving import RecommendationService
from glowguide.serving_artifacts import load_serving_bundle, save_serving_bundle
from serving_helpers import small_bundle


@pytest.mark.parametrize("different_content_weights", [False, True])
def test_lossless_compaction_and_persistence(tmp_path, different_content_weights):
    with patch("glowguide.serving_artifacts.compact_serving_bundle", side_effect=lambda b: b):
        source = small_bundle()
    if different_content_weights:
        # Duplicate positive rows are summed by content and maxed by CF.
        # Simulate that fitted state to ensure conversion never merges them.
        source.content._positive_weights.data[0] += 1.0
    before = joblib.hash(source)
    compact = compact_serving_bundle(source)
    assert joblib.hash(source) == before
    assert compact_serving_bundle(compact) is compact
    assert dict(compact.seen_by_user) == source.seen_by_user
    assert set(compact.positive_history_users) == source.positive_history_users
    assert dict(compact.profile._users) == source.profile._users
    assert (compact.content._positive_weights is compact.collaborative._positive_weights) is not different_content_weights
    assert compact.content._user_index.users.user_index is compact.collaborative._user_index
    assert compact.seen_by_user.user_index is compact.collaborative._user_index
    assert not hasattr(compact.profile, "user_profiles")
    assert not hasattr(compact.content, "vectorizer")
    assert not hasattr(compact.collaborative, "positive_matrix")
    path = tmp_path / "compact.joblib"
    save_serving_bundle(compact, path)
    loaded = load_serving_bundle(path)
    assert loaded.content._user_index.users.user_index is loaded.collaborative._user_index
    assert (loaded.content._positive_weights is loaded.collaborative._positive_weights) is not different_content_weights
    expected, actual = RecommendationService(source), RecommendationService(loaded)
    for user in (None, "unknown", "negative", "solo", "u", "v"):
        for name in ("popularity", "content", "profile", "collaborative"):
            np.testing.assert_array_equal(getattr(source, name).score_candidates(user_id=user),
                                          getattr(loaded, name).score_candidates(user_id=user))
        for filters in ({}, {"max_price": 20}, {"category": "serums", "in_stock_only": False}, {"skin_type": "dry"}):
            assert actual.recommend(user_id=user, **filters) == expected.recommend(user_id=user, **filters)
    with pytest.raises(ValueError, match="No positive training profile"):
        loaded.content.user_profile("negative")


@pytest.mark.parametrize("maximum,dtype", [(255, np.uint8), (256, np.uint16), (65535, np.uint16),
                                          (65536, np.uint32), (2**32, np.uint64)])
def test_integer_storage_never_overflows(maximum, dtype):
    assert unsigned_dtype(maximum) is dtype
