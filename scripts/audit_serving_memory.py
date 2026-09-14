"""Fresh-process RSS audit and exact serving response regression capture.

Run with --output artifacts/memory/before.json, then --compare that file.
psutil is diagnostic only; object sizes are recursive unique owned bytes,
not RSS. Inventory runs AFTER RSS sampling so traversal cannot inflate it.
"""

import argparse
import json
from pathlib import Path
import sys

import psutil


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
PROCESS = psutil.Process()
STAGES = []


def sample(stage):
    rss = PROCESS.memory_info().rss
    previous = STAGES[-1]["rss_bytes"] if STAGES else rss
    STAGES.append({"stage": stage, "rss_bytes": rss, "rss_mib": rss / 2**20,
                   "delta_mib": (rss - previous) / 2**20})
    print(json.dumps(STAGES[-1]), flush=True)


def owned_size(value, visited=None):
    """Count Python containers/attributes, pandas deep storage and numpy buffers.

    Separate field sizes overlap when fields share objects; the bundle total
    deduplicates objects by identity. Pandas reports its own deep storage.
    """
    import numpy as np
    import pandas as pd
    visited = set() if visited is None else visited
    if id(value) in visited:
        return 0
    visited.add(id(value))
    if isinstance(value, pd.DataFrame):
        return int(value.memory_usage(index=True, deep=True).sum())
    if isinstance(value, (pd.Series, pd.Index)):
        return int(value.memory_usage(deep=True))
    size = sys.getsizeof(value)
    if isinstance(value, np.ndarray):
        if value.base is not None:
            size += owned_size(value.base, visited)
        if value.dtype.hasobject:
            size += sum(owned_size(x, visited) for x in value.flat)
    elif isinstance(value, dict):
        size += sum(owned_size(k, visited) + owned_size(v, visited) for k, v in value.items())
    elif isinstance(value, (tuple, list, set, frozenset)):
        size += sum(owned_size(x, visited) for x in value)
    elif hasattr(value, "__dict__") and not isinstance(value, type):
        size += owned_size(vars(value), visited)
    return size


def capture(service, requests, products):
    return {"requests": requests, "responses": [service.recommend(**r) for r in requests],
            "products": products, "product_responses": [service.product(p) for p in products]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare", type=Path)
    args = parser.parse_args()
    sample("fresh Python (including psutil diagnostic)")
    import joblib
    import numpy as np
    from glowguide.serving import RecommendationService
    from glowguide.api.main import app  # Include the actual production imports.
    sample("after importing serving modules and API")
    path = ROOT / "artifacts/serving_bundle.joblib"
    bundle = joblib.load(path)
    sample("after joblib.load")
    service = RecommendationService(bundle)
    sample("after RecommendationService construction")
    user = next(iter(bundle.content._user_index))
    first = service.recommend(user_id=user)
    assert first["strategy"] == "collaborative"
    sample("after first collaborative request")
    service.recommend(skin_type="dry", skin_tone="light")
    sample("after first skin-profile request")
    service.recommend()
    sample("after first popularity request")
    report = {"platform": sys.platform, "stages": STAGES, "bundle_bytes": path.stat().st_size}
    if args.compare:
        baseline = json.loads(args.compare.read_text(encoding="utf-8"))
        cases = baseline["production"]
        report["production"] = capture(service, cases["requests"], cases["products"])
        assert report["production"] == cases, "Production serving responses changed"
    else:
        users = list(bundle.content._user_index)
        selected = [users[i] for i in np.linspace(0, len(users) - 1, 24, dtype=int)]
        negative = next(u for u in bundle.seen_by_user if u not in bundle.positive_history_users)
        requests = [{"user_id": u, "top_k": 20, "in_stock_only": False} for u in selected]
        requests += [{}, {"user_id": "unknown-user"}, {"user_id": negative},
                     {"user_id": negative, "skin_type": "dry"}]
        requests += [{"skin_type": t, "skin_tone": tone, "top_k": 50}
                     for t, tone in [("dry", "light"), (" Combination ", "light medium"),
                                     ("oily", None), (None, "deep"), ("unknown", "unknown")]]
        requests += [{**base, **filters} for base in ({}, {"skin_type": "dry"}, {"user_id": selected[0]})
                     for filters in ({"max_price": 25}, {"category": "  SERUMS "},
                                     {"in_stock_only": False}, {"max_price": 0, "category": "serums"},
                                     {"max_price": 50, "category": "moisturizers", "top_k": 1})]
        # Find real fallback candidates with no off-diagonal collaborative signal.
        weights = bundle.collaborative._positive_weights
        signal = np.diff(bundle.collaborative.item_similarity.indptr) > 0
        candidates = np.flatnonzero((np.asarray(weights @ signal).ravel() == 0) & (np.diff(weights.indptr) > 0))
        candidate_rows = set(candidates.tolist())
        reverse = {i: u for u, i in bundle.collaborative._user_index.items() if i in candidate_rows}
        fallback = next((u for u in reverse.values() if service.recommend(user_id=u)["strategy"] == "content_fallback"), None)
        if fallback is not None:
            requests.append({"user_id": fallback, "in_stock_only": False, "top_k": 50})
        report["production_content_fallback_found"] = fallback is not None
        products = list(bundle.product_metadata)
        products = [products[0], products[len(products)//2], products[-1], "unknown-product"]
        cold = sorted(set(bundle.product_metadata) - bundle.popularity.candidate_product_ids)
        products += cold[:1]
        report["production"] = capture(service, requests, products)
    from serving_helpers import small_bundle
    synthetic = RecommendationService(small_bundle())
    report["synthetic"] = capture(synthetic, [{"user_id": "solo"}, {"user_id": "u"},
        {"user_id": "negative"}, {"skin_type": "dry"}, {},
        {"user_id": "u", "category": "serums", "max_price": 20}], ["A", "COLD", "missing"])
    if args.compare:
        assert report["synthetic"] == baseline["synthetic"], "Synthetic serving responses changed"
        report["equivalence"] = "exact Python equality, including all scores and explanations"
    inventory = []
    for name, value in vars(bundle).items():
        if name in ("content", "profile", "collaborative", "popularity"):
            for field, obj in vars(value).items():
                inventory.append({"field": name + "." + field, "type": type(obj).__name__,
                                  "owned_mib": owned_size(obj) / 2**20})
        else:
            inventory.append({"field": name, "type": type(value).__name__, "owned_mib": owned_size(value) / 2**20})
    report["inventory"] = sorted(inventory, key=lambda x: -x["owned_mib"])
    report["bundle_owned_mib"] = owned_size(bundle) / 2**20
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("production", "synthetic")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
