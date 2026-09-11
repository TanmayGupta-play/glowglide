"""Synthetic serving routes, constraints, explanations and trusted persistence."""

from copy import deepcopy
import json
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from glowguide.api.schemas import RecommendationResponse
from glowguide.serving import RecommendationService
from glowguide.serving_artifacts import save_serving_bundle, load_serving_bundle, stock_flag, metadata_index
from serving_helpers import small_bundle


@pytest.fixture
def service():
    return RecommendationService(small_bundle())


def ids(result):
    return [p["product_id"] for p in result["recommendations"]]


def test_known_positive_routes_collaborative_excludes_positive_and_negative_seen(service):
    result = service.recommend(user_id="u", in_stock_only=False)
    assert result["strategy"] == "collaborative" and result["user_history_available"]
    assert set(ids(result)) == {"B", "C", "E", "F"}
    assert "A" not in ids(result) and "N" not in ids(result)
    assert result["candidate_pool_size"] == 4


def test_unavailable_collaborative_uses_content(service):
    assert not service.bundle.collaborative.score_candidates(user_id="solo").any()
    result = service.recommend(user_id="solo")
    assert result["strategy"] == "content_fallback" and result["returned_count"] > 0
    assert "D" not in ids(result)
    for item in result["recommendations"]:
        expected = service.bundle.content.top_matching_terms("solo", item["product_id"])
        assert item["explanation"]["matching_terms"] == [term for term, _ in expected]
        assert item["score_type"] == "tfidf_cosine_similarity"
        assert item["score"] == service.bundle.content.score_candidates(user_id="solo")[service._index[item["product_id"]]]


@pytest.mark.parametrize("profile", [{"skin_type": "Combination"}, {"skin_tone": "light medium"},
                                    {"skin_type": " Combination ", "skin_tone": "light medium"}])
def test_unknown_explicit_profile_route_and_explanation(service, profile):
    result = service.recommend(user_id="new", **profile)
    assert result["strategy"] == "skin_profile" and not result["user_history_available"]
    assert result["skin_profile_used"]["skin_type"] in (None, "combination")
    assert result["skin_profile_used"]["skin_tone"] in (None, "light_medium")
    scores = service.bundle.profile.score_explicit_profile(**profile)
    for item in result["recommendations"]:
        expected = service.bundle.profile.explain_explicit_profile(item["product_id"], **profile)
        assert item["explanation"] == {"type": "skin_profile", **expected}
        assert item["score"] == scores[service._index[item["product_id"]]]
        assert item["score_type"] == "smoothed_skin_profile_affinity"


def test_no_profile_popularity_and_negative_only_user_seen(service):
    anonymous = service.recommend()
    assert anonymous["strategy"] == "popularity"
    assert anonymous["skin_profile_used"] is None
    assert "A" in ids(anonymous)
    for item in anonymous["recommendations"]:
        assert item["score_type"] == "positive_user_count"
        assert item["score"] == item["explanation"]["positive_user_count"]
    negative = service.recommend(user_id="negative")
    assert not negative["user_history_available"] and "N" not in ids(negative)
    assert service.recommend(user_id="negative", skin_type="dry")["strategy"] == "skin_profile"


def test_both_personalized_signals_unavailable_follow_explicit_or_popularity(service):
    with patch.object(service.bundle.collaborative, "score_candidates", return_value=np.zeros(7)), patch.object(
        service.bundle.content, "score_candidates", return_value=np.zeros(7)
    ):
        assert service.recommend(user_id="u", skin_type="dry")["strategy"] == "skin_profile"
        assert service.recommend(user_id="u")["strategy"] == "popularity"


def test_filters_never_trigger_model_switch_or_change_scores(service):
    original = service.recommend(user_id="u", in_stock_only=False)
    by_id = {p["product_id"]: p for p in original["recommendations"]}
    assert "C" in by_id and "C" not in ids(service.recommend(user_id="u"))
    budget = service.recommend(user_id="u", max_price=20, in_stock_only=False)
    assert set(ids(budget)) == {"B", "F"}  # Missing E price fails explicit budget.
    category = service.recommend(user_id="u", category="  FACE   SERUMS  ")
    assert ids(category) == ["F"]
    # F is below the unfiltered top one: filtering must search beyond top_k.
    assert ids(service.recommend(user_id="u", category="face serums", top_k=1)) == ["F"]
    combined = service.recommend(user_id="u", category="serums", max_price=20)
    assert ids(combined) == ["B"]
    assert combined["recommendations"][0] == by_id["B"]
    assert combined["returned_count"] < combined["requested_top_k"]
    empty = service.recommend(user_id="u", max_price=0, category="serums")
    assert empty["strategy"] == "collaborative" and empty["returned_count"] == 0
    assert empty["candidate_pool_size"] == original["candidate_pool_size"]


def test_collaborative_explanations_and_raw_scores(service):
    result = service.recommend(user_id="u")
    for item in result["recommendations"]:
        source = service.bundle.collaborative.explain_score("u", item["product_id"], top_n=3)
        explanation = item["explanation"]
        assert item["score_type"] == "adjusted_collaborative_affinity"
        assert item["score"] == pytest.approx(source["final_score"])
        assert explanation["other_score_contribution"] == source["other_score_contribution"]
        assert len(explanation["because_you_liked"]) <= 3
        for actual, expected in zip(explanation["because_you_liked"], source["contributions"]):
            assert actual["product_id"] == expected["history_product_id"]
            assert actual["product_name"] == service.product(actual["product_id"])["product_name"]
            assert actual["similarity"] == expected["adjusted_similarity"]


def test_determinism_no_nan_no_mutation(service):
    bundle_hash = joblib.hash(service.bundle)
    request = {"user_id": "u", "in_stock_only": False, "top_k": 50}
    before = deepcopy(request)
    result = service.recommend(**request)
    assert result == service.recommend(**request)
    assert ids(result) == ["B", "C", "E", "F"]  # Exact collaborative ties.
    assert len(ids(result)) == len(set(ids(result)))
    json.dumps(result, allow_nan=False)
    RecommendationResponse.model_validate(result)
    assert all(p["rating"] is None for p in result["recommendations"])
    assert before == request and joblib.hash(service.bundle) == bundle_hash
    product = service.product("B")
    product["product_name"] = "changed"
    assert service.product("B")["product_name"] != "changed"


def test_explicit_profile_all_missing_and_partial_no_mutation(service):
    model = service.bundle.profile
    before = joblib.hash(model)
    np.testing.assert_array_equal(model.score_explicit_profile(), model.score_candidates(user_id="unknown"))
    np.testing.assert_array_equal(model.score_explicit_profile("Combination", "light medium"), model.score_candidates(user_id="u"))
    for skin_type, skin_tone in [(None, None), ("Combination", None), (None, "light medium"), ("Combination", "light medium")]:
        scores = model.score_explicit_profile(skin_type, skin_tone)
        for i, product_id in enumerate(model.product_ids):
            assert scores[i] == model.explain_explicit_profile(product_id, skin_type, skin_tone)["final_score"]
    assert joblib.hash(model) == before


def test_trusted_bundle_roundtrip_metadata_and_behavior(service, tmp_path):
    path = tmp_path / "bundle.joblib"
    save_serving_bundle(service.bundle, path)
    loaded = load_serving_bundle(path)
    assert loaded.build_metadata == service.bundle.build_metadata
    assert loaded.seen_by_user == service.bundle.seen_by_user
    other = RecommendationService(loaded)
    for request in ({}, {"user_id": "u"}, {"user_id": "solo"}, {"skin_type": "dry"}):
        assert other.recommend(**request) == service.recommend(**request)
    assert loaded.build_metadata["interaction_rows"] == 10
    assert loaded.build_metadata["historical_users"] == 4
    assert loaded.build_metadata["candidate_products"] == 7
    assert loaded.build_metadata["full_skincare_catalog_products"] == 8
    assert loaded.build_metadata["catalog_products_without_historical_interactions"] == 1
    assert "COLD" not in ids(other.recommend()) and other.product("COLD") is not None


def test_missing_corrupt_incompatible_bundle_errors(service, tmp_path):
    with pytest.raises(FileNotFoundError, match="build_serving_bundle"):
        load_serving_bundle(tmp_path / "absent")
    path = tmp_path / "corrupt"
    path.write_bytes(b"not a joblib bundle")
    with pytest.raises(RuntimeError, match="corrupt or incompatible"):
        load_serving_bundle(path)
    service.bundle.build_metadata["bundle_version"] = "unsupported"
    joblib.dump(service.bundle, path)
    with pytest.raises(RuntimeError, match="version"):
        load_serving_bundle(path)


@pytest.mark.parametrize("value,expected", [(0, False), (1, True), ("0.0", False), ("1.0", True),
                                           ("true", True), ("FALSE", False), (None, None), (np.nan, None), ("unknown", None)])
def test_stock_dtype(value, expected):
    assert stock_flag(value) is expected


def test_optional_metadata_is_safe():
    frame = pd.DataFrame({"product_id": ["A"], "price_usd": [np.inf], "rating": [np.nan]})
    before = frame.copy(deep=True)
    index = metadata_index(frame, {"A", "B"})
    assert index["A"]["price_usd"] is None and index["B"]["product_name"] is None
    json.dumps(index, allow_nan=False)
    assert_frame_equal(frame, before)


@pytest.mark.parametrize("payload", [{"top_k": 0}, {"top_k": 51}, {"top_k": True}, {"max_price": -1},
                                     {"max_price": np.inf}, {"max_price": np.nan}])
def test_invalid_service_requests(service, payload):
    with pytest.raises(ValueError):
        service.recommend(**payload)
