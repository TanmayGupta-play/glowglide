"""Deployment smoke checks detect service failures without loading ML models."""

from copy import deepcopy
import json

import httpx
import pytest

from scripts.verify_deployment import VerificationError, api_url, main, verify


def responses():
    return {
        "health": {"status": "ok", "bundle_loaded": True, "bundle_version": "1", "candidate_products": 2},
        **{strategy: {
            "strategy": strategy, "user_history_available": False, "bundle_version": "1",
            "skin_profile_used": {"skin_type": "dry", "skin_tone": "light"} if strategy == "skin_profile" else None,
            "requested_top_k": 5, "returned_count": 1,
            "recommendations": [{"product_id": "product-1", "score": 1.0, "explanation": {"type": strategy}}],
        } for strategy in ("popularity", "skin_profile")},
        "product": {"product_id": "product-1"},
    }


def transport(bodies, requests):
    def handle(request):
        requests.append(request)
        if request.url.path == "/api/health":
            key = "health"
        elif request.url.path == "/api/recommendations":
            body = json.loads(request.content)
            assert "user_id" not in body
            key = "skin_profile" if "skin_type" in body else "popularity"
        else:
            assert request.url.path == "/api/products/product-1"
            key = "product"
        return httpx.Response(200, json=bodies[key])
    return httpx.MockTransport(handle)


def test_success_uses_both_anonymous_routes_and_product_lookup():
    requests = []
    with httpx.Client(base_url=api_url("https://example.test/api"), transport=transport(responses(), requests)) as client:
        summary = verify(client)
    assert "health OK" in summary and "product lookup OK" in summary
    assert [request.method for request in requests] == ["GET", "POST", "POST", "GET"]


@pytest.mark.parametrize("section,field,value", [
    ("health", "bundle_loaded", False),
    ("health", "candidate_products", 0),
    ("popularity", "recommendations", []),
    ("popularity", "returned_count", 2),
    ("skin_profile", "strategy", "popularity"),
    ("skin_profile", "bundle_version", "other"),
    ("product", "product_id", "other-product"),
])
def test_invalid_service_results_fail(section, field, value):
    bodies = deepcopy(responses())
    bodies[section][field] = value
    with httpx.Client(base_url="https://example.test/api/", transport=transport(bodies, [])) as client:
        with pytest.raises(VerificationError):
            verify(client)


@pytest.mark.parametrize("response", [
    httpx.Response(503, text="private response body"),
    httpx.Response(200, text="not JSON"),
    httpx.Response(200, json=[]),
])
def test_http_and_json_errors_fail_without_response_body(response):
    with httpx.Client(base_url="https://example.test/", transport=httpx.MockTransport(lambda request: response)) as client:
        with pytest.raises(VerificationError) as error:
            verify(client)
    assert "private response body" not in str(error.value)


def test_cli_connection_failure_returns_nonzero_without_private_details(monkeypatch, capsys):
    def fail(client):
        raise VerificationError("GET request failed: connection, TLS, or timeout error")
    monkeypatch.setattr("scripts.verify_deployment.verify", fail)
    assert main(["--api-url", "http://localhost:8000"]) == 1
    assert "Deployment verification failed" in capsys.readouterr().err
