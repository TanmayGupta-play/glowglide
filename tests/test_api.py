"""FastAPI tests inject a tiny service; no real serving bundle is required."""

import json
from unittest.mock import patch

from fastapi.testclient import TestClient
import pytest

from glowguide.api.main import create_app
from glowguide.serving import RecommendationService
from glowguide.serving_artifacts import save_serving_bundle
from serving_helpers import small_bundle


@pytest.fixture
def service():
    return RecommendationService(small_bundle())


@pytest.fixture
def client(service):
    with TestClient(create_app(service)) as client:
        yield client


def test_health_and_recommendations(client):
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json() == {"status": "ok", "bundle_loaded": True, "bundle_version": "1", "candidate_products": 7, "historical_users": 4}
    response = client.post("/recommendations", json={"user_id": "u"})
    assert response.status_code == 200 and response.json()["strategy"] == "collaborative"
    json.dumps(response.json(), allow_nan=False)


@pytest.mark.parametrize("payload", [{"top_k": 0}, {"top_k": 51}, {"top_k": 1.5}, {"top_k": True},
                                     {"max_price": -1}, {"max_price": "NaN"}, {"max_price": "Infinity"}, {"unexpected": "value"}])
def test_request_validation(client, payload):
    assert client.post("/recommendations", json=payload).status_code == 422


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity"])
def test_nonstandard_nonfinite_json_is_422_not_500(client, value):
    response = client.post("/recommendations", content='{"max_price":' + value + '}', headers={"Content-Type": "application/json"})
    assert response.status_code == 422
    json.dumps(response.json(), allow_nan=False)


def test_products(client):
    assert client.get("/products/missing").status_code == 404
    result = client.get("/products/B")
    assert result.status_code == 200
    assert result.json()["price_usd"] == 20
    assert result.json()["rating"] is None
    assert client.get("/products/COLD").status_code == 200


def test_no_matches_is_success(client):
    response = client.post("/recommendations", json={"user_id": "u", "max_price": 0, "category": "serums"})
    assert response.status_code == 200
    assert response.json()["recommendations"] == [] and response.json()["returned_count"] == 0
    assert response.json()["strategy"] == "collaborative"


def test_profile_normalization_and_optional_user(client):
    response = client.post("/recommendations", json={"user_id": " ", "skin_type": " Combination ", "skin_tone": "light medium"})
    assert response.status_code == 200
    result = response.json()
    assert result["strategy"] == "skin_profile"
    assert result["skin_profile_used"] == {"skin_type": "combination", "skin_tone": "light_medium"}
    assert client.post("/recommendations", json={"skin_type": " ", "skin_tone": ""}).json()["strategy"] == "popularity"
    assert client.post("/recommendations", json={"user_id": "solo"}).json()["strategy"] == "content_fallback"


def test_lifespan_loads_once_from_override(service, tmp_path, monkeypatch):
    path = tmp_path / "serving.joblib"
    save_serving_bundle(service.bundle, path)
    monkeypatch.setenv("GLOWGUIDE_BUNDLE_PATH", str(path))
    from glowguide.api import main
    with patch.object(main, "load_serving_bundle", wraps=main.load_serving_bundle) as load:
        app = create_app()
        assert load.call_count == 0
        with TestClient(app) as client:
            client.get("/health")
            client.post("/recommendations", json={})
            client.post("/recommendations", json={"user_id": "u"})
            assert load.call_count == 1


def test_startup_missing_bundle_fails(tmp_path, monkeypatch):
    monkeypatch.setenv("GLOWGUIDE_BUNDLE_PATH", str(tmp_path / "missing.joblib"))
    with pytest.raises(FileNotFoundError, match="Serving bundle missing"), TestClient(create_app()):
        pass


def test_default_path_independent_of_cwd(service, tmp_path, monkeypatch):
    from glowguide.api import main
    monkeypatch.delenv("GLOWGUIDE_BUNDLE_PATH", raising=False)
    monkeypatch.chdir(tmp_path)
    with patch.object(main, "load_serving_bundle", return_value=service.bundle) as load, TestClient(create_app()):
        load.assert_called_once_with(main.PROJECT_ROOT / "artifacts" / "serving_bundle.joblib")


def test_cors_config(service, monkeypatch):
    monkeypatch.setenv("GLOWGUIDE_CORS_ORIGINS", "http://localhost:3000, https://example.test")
    with TestClient(create_app(service)) as client:
        response = client.options("/recommendations", headers={"Origin": "https://example.test", "Access-Control-Request-Method": "POST"})
        assert response.status_code == 200
        assert response.headers["access-control-allow-origin"] == "https://example.test"
        assert response.headers.get("access-control-allow-credentials") != "true"
