"""Run INSIDE a live serving container; check HTTP and Linux cgroup memory.

Example: docker cp this script and the Linux baseline into /tmp, then
docker exec <container> python /tmp/verify_container_memory.py /tmp/before-linux.json
The probe imports no serving models and records its own memory within cgroup.
"""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
from time import perf_counter
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

import psutil


CGROUP = Path("/sys/fs/cgroup")


def memory():
    return {name: int((CGROUP / name).read_text()) for name in ("memory.current", "memory.peak", "memory.max")}


def request(path, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    with urlopen(Request("http://127.0.0.1:8000/" + path, data=data,
                         headers={"Content-Type": "application/json"}), timeout=60) as response:
        assert response.status == 200
        return json.load(response)


def main():
    start = perf_counter()
    baseline = json.loads(Path(sys.argv[1]).read_text())["production"]
    snapshots = [{"stage": "before HTTP checks", **memory()}]
    assert memory()["memory.max"] == 512 * 2**20
    health = request("health")
    assert health["status"] == "ok" and health["bundle_loaded"]
    cases = list(zip(baseline["requests"], baseline["responses"]))

    def check(case):
        payload, expected = case
        actual = request("recommendations", payload)
        assert actual == expected, "HTTP serving response changed"
        return actual["strategy"]

    for case in cases:
        strategy = check(case)
        snapshots.append({"stage": strategy, **memory()})
    for product, expected in zip(baseline["products"], baseline["product_responses"]):
        try:
            assert request("products/" + quote(product, safe="")) == expected
        except HTTPError as exc:
            assert expected is None and exc.code == 404
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(check, cases * 3))
    snapshots.append({"stage": "after 4 concurrent clients", **memory()})
    assert request("health") == health
    events = dict(line.split() for line in (CGROUP / "memory.events").read_text().splitlines())
    assert int(events["oom"]) == 0 and int(events["oom_kill"]) == 0
    assert memory()["memory.peak"] < memory()["memory.max"]
    report = {"result": "passed", "health": health, "http_equivalence": "exact",
              "recommendation_requests": len(cases) * 4, "product_lookups": len(baseline["products"]),
              "strategies": sorted({expected["strategy"] for _, expected in cases}),
              "concurrent_clients": 4, "uvicorn_rss_mib": psutil.Process(1).memory_info().rss / 2**20,
              "probe_rss_mib": psutil.Process().memory_info().rss / 2**20,
              "memory_events": events, "snapshots": snapshots,
              "duration_seconds": perf_counter() - start}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
