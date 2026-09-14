# Serving memory audit — 2026-09-14

The existing production backend fits the tested 512 MiB container limit after
lossless serving storage compaction. No algorithms, tuning, evaluation,
frontend behavior, production deployment, commits, or pushes were changed.

**Process RSS measurements.** Each audit ran in a fresh Python process, using
the already-installed psutil. No diagnostic dependency was added. The import
stage includes the production FastAPI module and its dependencies. Deltas are
relative to the preceding stage. Inventory traversal and regression capture
run only after the seven RSS measurements. Values below are MiB, rounded to
five decimal places; JSON evidence retains exact byte counts.

Linux, using the production Docker image and dependency versions:

| Stage | Before RSS | Delta | After RSS | Delta |
|---|---:|---:|---:|---:|
| Fresh Python, including psutil | 18.10547 | 0.00000 | 18.45313 | 0.00000 |
| Serving/API imports | 156.28125 | 138.17578 | 157.35547 | 138.90234 |
| joblib.load | 643.84766 | 487.56641 | 284.14844 | 126.79297 |
| RecommendationService construction | 643.84766 | 0.00000 | 284.14844 | 0.00000 |
| First collaborative request | 644.22266 | 0.37500 | 284.64844 | 0.50000 |
| First skin-profile request | 644.22266 | 0.00000 | 284.64844 | 0.00000 |
| First popularity request | 644.22266 | 0.00000 | 284.64844 | 0.00000 |

Windows, using the existing project virtual environment:

| Stage | Before RSS | Delta | After RSS | Delta |
|---|---:|---:|---:|---:|
| Fresh Python, including psutil | 18.92969 | 0.00000 | 18.89063 | 0.00000 |
| Serving/API imports | 147.00781 | 128.07813 | 146.85156 | 127.96094 |
| joblib.load | 616.01953 | 469.01172 | 274.66797 | 127.81641 |
| RecommendationService construction | 616.10938 | 0.08984 | 274.77734 | 0.10938 |
| First collaborative request | 616.21484 | 0.10547 | 274.88672 | 0.10938 |
| First skin-profile request | 616.21484 | 0.00000 | 274.89063 | 0.00391 |
| First popularity request | 616.21484 | 0.00000 | 274.89063 | 0.00000 |

**Measured object inventory and changes.** Field sizes below are recursive
Python/numpy owned-storage measurements; pandas uses deep memory_usage.
Standalone field sizes overlap where they reference the same objects and
must not be added together. They are distinct from process RSS.

| Original field | MiB | Serving storage change |
|---|---:|---|
| seen_by_user | 171.33 | uint32 offsets and uint16 product indexes; only the requested user's frozenset is materialized |
| profile.user_profiles DataFrame | 94.08 | Removed fit-only DataFrame; inferred values retained in compact mapping |
| profile._users | 69.98 | Shared user index, uint8 profile codes, small table of unique profile tuples |
| collaborative._user_index | 56.54 | Retained as the single shared user-to-row dictionary |
| content._user_index | 50.45 | Mapping view over the shared dictionary and positive-user mask |
| positive_history_users | 40.30 | Boolean mask over the shared user index |
| collaborative.positive_matrix | 12.15 | Removed fit-only binary matrix; learned similarities retained exactly |
| collaborative._positive_weights | 12.15 | Retained unchanged as CSR |
| content._positive_weights | 11.87 | Aligned to shared row space and shared with collaborative after exact equality check |
| collaborative.item_similarity | 11.61 | Retained unchanged as CSR, including float64 values |
| content.vectorizer | 6.53 | Removed fit-only vectorizer; fitted vectors, transpose and terms retained |
| product_metadata | 1.34 | Retained unchanged |

The original bundle did not retain a raw interaction DataFrame or a separate
content dict of user-history sets. Its duplicated history storage was the
seen-set dictionary and the positive sparse matrices. The retained pandas
DataFrame was the inferred profile table. There was one product metadata
dictionary, not multiple metadata DataFrames. Product tuples, candidate sets,
and product-index dictionaries are now shared across the fitted serving models.

After compaction, the seen-history arrays occupy exactly 4,190,640 bytes;
profile codes and the positive-user mask each occupy 503,216 bytes. All refer
to the same 56.54 MiB user dictionary, which is now the largest object.
The recursive bundle storage estimate decreases from 451.54 to 104.80 MiB.
The deployed request path retains the existing scoring and explanation methods.

**Artifact rebuild.** Ran `python scripts/build_serving_bundle.py`, fitting
the unchanged Models 0–3 on all 1,088,886 processed interactions. Rebuilt both
`artifacts/serving_bundle.joblib` and `artifacts/serving_manifest.json`.
All 503,216 users, 2,351 ranked candidates, and 2,420 metadata products remain.

| Artifact | Before | After |
|---|---:|---:|
| serving_bundle.joblib bytes | 37,945,012 | 24,160,226 |
| serving_bundle.joblib MiB | 36.18718 | 23.04099 |

The API bundle version remains `1`, preserving response values. The manifest
adds `serving_representation: compact-v1`. Original artifacts are backed up
under `artifacts/memory/` for comparison.

**Behavior preservation.** Saved baseline responses before changing serving
storage. Exact Python equality passed for 49 production recommendation cases
and six synthetic cases on both Windows and Linux. The cases include 24
deterministically spaced collaborative users, a real content-fallback user,
negative-only and unknown users, full/partial/unknown skin profiles, popularity,
stock/category/price filters, empty results, and product lookups including cold
catalog and missing IDs. Strategy, product order, scores, score_type,
explanations, filtering, and all other response fields compare exactly; no
floating tolerance was needed.

The exhaustive storage comparison also passed for all 503,216 users,
1,088,886 seen pairs, 430,091 positive user rows, inferred profiles, 2,420
metadata records, and every retained learned score array. Content vectors,
terms, similarities, profile statistics, weights, and popularity scores are
exactly equal. Content and collaborative weights are shared only after exact
CSR equality; a regression test verifies that differing weights, such as those
produced by duplicate positive input rows, stay separate. Source fitted objects
are not mutated, and shared storage survives joblib round trips.

**Container verification.** Rebuilt `glowguide-memory-optimized` from the
unchanged Dockerfile and ran:

```powershell
docker build -t glowguide-memory-optimized .
docker run -d --name glowguide-memory-512 --memory=512m --memory-swap=512m -p 127.0.0.1:18000:8000 glowguide-memory-optimized
docker stats --no-stream glowguide-memory-512
```

The recorded HTTP probe passed `/health`, 196 recommendation requests across
all four strategies, and five product lookups, comparing complete JSON responses
to the Linux baseline. This includes a run with four concurrent clients.
Container inspection reported `healthy`, `Running: true`, `OOMKilled: false`.
All cgroup memory-event counters were zero, including `max`, `oom`, and
`oom_kill`. The hard limit was exactly 536,870,912 bytes, with no extra swap.

| Measurement | MiB |
|---|---:|
| Docker stats after probe exited | 251.1 / 512 |
| Recorded cgroup memory.peak, including probe and startup | 281.171875 |
| Uvicorn process RSS after concurrent requests | 292.72265625 |
| Diagnostic process RSS, charged to the same cgroup | 27.66796875 |

RSS, Docker stats, and cgroup usage have different accounting of shared pages
and cache; these are separately measured values, not interchangeable totals.
The local test container was stopped after verification. No deployment occurred.

**Tests and frozen metrics.** `python -m pytest -q` completed with **153 passed,
84 subtests passed**, and two existing dependency deprecation warnings, in
7.60 seconds. `git diff --check` passed. Models 0–4 recommender source,
preprocessing, evaluation, tuning, and frozen metric files were not modified.
Full frozen benchmarks were not rerun: the requested trigger, a shared
recommender-code change, did not occur. Existing model tests passed as part
of the full suite; exhaustive learned-array comparison supplies additional
serving rebuild evidence without changing evaluation.

**Files changed.**

- `src/glowguide/serving_artifacts.py`: compact the fitted serving copies and accept compact Mapping/Set interfaces.
- `src/glowguide/compact_serving.py`: lossless storage adapters and conversion.
- `scripts/audit_serving_memory.py`: staged RSS, object inventory, deterministic response capture/comparison.
- `scripts/verify_serving_storage.py`: exhaustive artifact representation comparison.
- `scripts/verify_container_memory.py`: HTTP equivalence, concurrency, RSS and cgroup verification.
- `tests/test_compact_serving.py`: exact behavior, serialization sharing, divergent weights, and integer capacity checks.
- `docs/SERVING_MEMORY_AUDIT.md`: this report.
- `artifacts/serving_bundle.joblib`, `artifacts/serving_manifest.json`: rebuilt, ignored by Git as before.

Raw evidence is in `artifacts/memory/before.json`, `after.json`,
`before-linux.json`, `after-linux.json`, `storage-equivalence.json`, and
`container.json`. Baselines contain historical user IDs and stay in the
existing ignored artifacts directory.

**Remaining limits.** This validates the existing single-worker production
command and four concurrent clients, not an extended production soak or
arbitrary worker counts. Additional workers would load additional bundles.
Artifact fitting/rebuilding remains an offline operation and is not expected
to fit 512 MiB. Render itself was not tested or deployed; the local Linux
container with the requested hard limit passed. Use the rebuilt bundle with
the new compact-serving module when deploying later.

**Git status.** One tracked source file modified (`serving_artifacts.py`),
six new source/test/script/report files untracked as listed above, and the
pre-existing untracked `eda_output.txt` left untouched. Artifacts and audit
JSON are ignored. Nothing staged, committed, or pushed.
