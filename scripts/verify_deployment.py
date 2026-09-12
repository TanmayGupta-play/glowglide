#!/usr/bin/env python3
"""Smoke-check an HTTP deployment without historical users or local ML data."""

import argparse
import math
import sys
from urllib.parse import quote, urlsplit

import httpx


class VerificationError(Exception):
    """A deployment did not satisfy the smoke-check contract."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def request_json(client: httpx.Client, method: str, path: str, **kwargs) -> dict:
    try:
        response = client.request(method, path, **kwargs)
    except httpx.HTTPError:
        # Do not print remote response bodies, URLs with credentials, or user data.
        raise VerificationError(f"{method} request failed: connection, TLS, or timeout error") from None
    require(response.status_code == 200, f"{method} request returned HTTP {response.status_code}; expected 200")
    try:
        body = response.json()
    except ValueError:
        raise VerificationError(f"{method} response was not valid JSON") from None
    require(isinstance(body, dict), f"{method} response must be a JSON object")
    return body


def verify(client: httpx.Client) -> str:
    health = request_json(client, "GET", "health")
    require(health.get("status") == "ok" and health.get("bundle_loaded") is True,
            "Health check did not confirm a loaded bundle")
    version = health.get("bundle_version")
    require(isinstance(version, str) and bool(version), "Health response has no bundle version")
    require(type(health.get("candidate_products")) is int and health["candidate_products"] > 0,
            "Health response has no candidate products")

    counts = []
    product_id = None
    for strategy, profile in (("popularity", {}), ("skin_profile", {"skin_type": "dry", "skin_tone": "light"})):
        body = request_json(client, "POST", "recommendations", json={"top_k": 5, **profile})
        require(body.get("strategy") == strategy, f"Expected {strategy} recommendation strategy")
        require(body.get("user_history_available") is False, "Anonymous request unexpectedly used user history")
        require(body.get("bundle_version") == version, "Recommendation bundle version differs from health")
        expected_profile = profile or None
        require(body.get("skin_profile_used") == expected_profile, f"Unexpected profile for {strategy}")
        items = body.get("recommendations")
        require(isinstance(items, list) and 0 < len(items) <= 5, f"{strategy} must return 1-5 recommendations")
        require(type(body.get("returned_count")) is int and body["returned_count"] == len(items)
                and body.get("requested_top_k") == 5, f"{strategy} recommendation counts are inconsistent")
        ids = set()
        for item in items:
            require(isinstance(item, dict), f"Invalid {strategy} recommendation item")
            item_id = item.get("product_id")
            require(isinstance(item_id, str) and bool(item_id.strip()), "Recommendation has no product ID")
            require(item_id not in ids, "Recommendations contain duplicate products")
            ids.add(item_id)
            score = item.get("score")
            require(type(score) in (int, float) and math.isfinite(score), "Recommendation has an invalid score")
            explanation = item.get("explanation")
            require(isinstance(explanation, dict) and explanation.get("type") == strategy,
                    f"Missing {strategy} explanation")
        product_id = items[0]["product_id"]
        counts.append(len(items))

    product = request_json(client, "GET", f"products/{quote(product_id, safe='')}")
    require(product.get("product_id") == product_id, "Product lookup returned a different product")
    return (f"Deployment verified: health OK, bundle loaded, popularity {counts[0]} products, "
            f"skin profile {counts[1]} products, product lookup OK.")


def api_url(value: str) -> str:
    try:
        parsed = urlsplit(value)
        valid = (parsed.scheme in ("http", "https") and parsed.hostname and parsed.port != 0
                 and parsed.username is None and parsed.password is None
                 and not parsed.query and not parsed.fragment)
    except ValueError:
        valid = False
    if not valid:
        raise argparse.ArgumentTypeError("Use an HTTP(S) API URL without credentials, query, or fragment")
    return value.rstrip("/") + "/"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", required=True, type=api_url, help="API base URL, e.g. http://localhost:8000")
    args = parser.parse_args(argv)
    try:
        with httpx.Client(base_url=args.api_url, timeout=30.0) as client:
            summary = verify(client)
    except VerificationError as exc:
        print(f"Deployment verification failed: {exc}", file=sys.stderr)
        return 1
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
