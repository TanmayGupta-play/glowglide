# Engineering Handover

[Overview](../README.md) · [API](API.md) · [Architecture](ARCHITECTURE.md) · [Limitations](LIMITATIONS.md)

## System prerequisites

Verified environment: **Windows, Python 3.11.8, Node.js 22.19.0, npm 10.9.3**. Python dependencies are pinned in [requirements.txt](../requirements.txt); frontend versions are pinned by [package.json](../frontend/package.json) and its lockfile. Linux commands are provided for portability but were not executed in this handover. There is no deployment setup or minimum-memory guarantee.

From the repository root:

```sh
python -m venv .venv
```

PowerShell activation:

```powershell
.venv\Scripts\Activate.ps1
```

Bash activation:

```bash
source .venv/bin/activate
```

Then install:

```sh
python -m pip install -r requirements.txt
```

Keep the same environment for bundle building and API startup. Compatibility checks compare exact Python, NumPy, pandas, SciPy, scikit-learn and joblib versions, including Python's patch version. Rebuild locally after changing these dependencies.

## Required data

Obtain the historical `nadyinky/sephora-products-and-skincare-reviews` dataset. The repository's optional download helper is:

```sh
python scripts/download_data.py
```

It uses kagglehub and copies downloaded CSVs into `data/raw/`. It requires external dataset/network access; this handover did not redownload the data. Follow the dataset provider's access and licensing terms.

The local source layout used here is:

```text
data/raw/
  product_info.csv
  reviews_0-250.csv
  reviews_250-500.csv
  reviews_500-750.csv
  reviews_750-1250.csv
  reviews_1250-end.csv
```

The loader specifically requires `product_info.csv` and at least one `reviews_*.csv`, loading all matching review files in sorted filename order. A partial review download may run but will not reproduce the frozen counts. Product required columns: `product_id`, `primary_category`. Review required columns: `author_id`, `product_id`, `rating`, `submission_time`. Other supported metadata/profile fields are optional. Do not leave duplicate copies under the review filename pattern.

Raw data, processed data and `artifacts/` contents are ignored by Git. They are not supplied merely by cloning source. Preserve required local data/artifacts through an appropriate trusted handover rather than assuming they are committed.

## Building processed data

```sh
python scripts/build_processed_data.py
```

Outputs (overwritten on rebuild):

- `data/processed/products_skincare.csv`
- `data/processed/interactions.csv`

The script prints row accounting, user/product counts, label totals, date range and profile coverage. It validates ratings/dates/IDs, restricts to skincare, normalizes profile values and keeps the latest review per user-product pair. Existing semantics are frozen; do not “clean up” this logic while reproducing experiments.

Current expected outputs: 2,420 skincare products and 1,088,886 interactions from 503,216 users. Review `scripts/eda_summary.py`, `scripts/inspect_data.py` and `scripts/test_setup.py` for existing data inspection helpers. `eda_output.txt` is unrelated local output and should be left untouched.

## Building serving bundle

```sh
python scripts/build_serving_bundle.py
```

Outputs:

- `artifacts/serving_bundle.joblib`
- `artifacts/serving_manifest.json`

This deliberately fits Models 0–3 on **all processed history** after model selection, with collaborative shrinkage fixed at 10. It does not train Model 4, split data for serving, or overwrite frozen benchmark JSON. The manifest records counts, timestamp range, model/vectorizer/smoothing configuration, dependency versions, UTC build time, size and runtime.

**Load joblib artifacts only from trusted sources.** Pickle-style deserialization can execute code before subsequent compatibility checks. Never accept bundles through user uploads or untrusted download links. Version validation is not a security sandbox.

A rebuild replaces local serving artifacts. Restart the API after a successful rebuild; the already-running process retains its in-memory bundle. For support work, preserve a trusted prior artifact and its matching environment if rollback is needed. There is no automated registry or rollback workflow.

## Starting API

```sh
python -m uvicorn glowguide.api.main:app --app-dir src --reload
```

Run from the repository root. `--app-dir src` makes the source layout importable; `--reload` is a local development convenience, not deployment configuration.

Defaults: API port 8000, trusted bundle `artifacts/serving_bundle.joblib`, allowed browser origin `http://localhost:3000`. Configuration examples:

```powershell
$env:GLOWGUIDE_BUNDLE_PATH = "artifacts/serving_bundle.joblib"
$env:GLOWGUIDE_CORS_ORIGINS = "http://localhost:3000"
```

```bash
export GLOWGUIDE_BUNDLE_PATH=artifacts/serving_bundle.joblib
export GLOWGUIDE_CORS_ORIGINS=http://localhost:3000
```

Relative bundle overrides resolve from the project root. CORS accepts a comma-separated origin list and does not allow credentials. See [API configuration](API.md#configuration-and-startup).

Open `http://localhost:8000/health` or use this Python command in another terminal:

```sh
python -c "import urllib.request; print(urllib.request.urlopen('http://localhost:8000/health').read().decode())"
```

Expect `status: ok`, `bundle_loaded: true`, version `1`, 2,351 candidate products and 503,216 historical users with the current data. Swagger is `http://localhost:8000/docs`.

## Starting frontend

In another terminal:

```sh
cd frontend
npm ci
npm run dev
```

Open `http://localhost:3000`. Optional local configuration: copy `.env.example` to `.env.local` inside `frontend/` and set:

```dotenv
NEXT_PUBLIC_GLOWGUIDE_API_URL=http://localhost:8000
```

The fallback is the same localhost URL. Restart the development server after environment changes; a production frontend build captures public environment configuration and must be rebuilt for changes. Never place credentials in a `NEXT_PUBLIC_` variable.

Use no user ID for a profile-only or popularity demo. An existing authorized test identifier can be entered manually; do not publish real historical IDs in docs, screenshots or client fixtures.

## Common maintenance tasks

| Task | Procedure |
|---|---|
| Refresh source data | Obtain the intended CSV snapshot; keep provenance. Rebuild processed data and inspect counts before rebuilding serving artifacts. |
| Refresh recommendations | Rebuild the serving bundle, verify its manifest, restart API and check health/recommendations. |
| Inspect EDA | Run the existing inspection/EDA scripts against local raw files; do not overwrite unrelated saved output. |
| Reproduce frozen experiments | Use the original processed snapshot and commands below. Preserve old artifacts before an intentional rerun, since scripts write their metric JSON. |
| Verify backend | `python -m pytest -q` from root. Synthetic fixtures avoid loading the million-row dataset. |
| Verify frontend | `npm run lint`, `npm test`, `npm run build` from `frontend/`. Tests mock API responses. |

Experiment commands, from the root:

```sh
python scripts/run_baseline.py
python scripts/run_content_baseline.py
python scripts/run_profile_baseline.py
python scripts/run_collaborative_baseline.py
python scripts/run_hybrid.py
```

`run_hybrid.py` performs nested tuning and then the frozen outer evaluation; it is not necessary to start or rebuild production serving. Do not manually select fusion weights after inspecting outer results. Do not regenerate frozen benchmarks on a changed dataset and label them as the original experiment.

## Troubleshooting

| Symptom | Check / action |
|---|---|
| Missing raw CSVs | Confirm `data/raw/product_info.csv` and all source review shards; the loader raises clear file/required-column errors. |
| Missing processed data | Run preprocessing after placing raw CSVs. Do not expect ignored processed files in a clean clone. |
| Missing serving bundle | Run the bundle builder in the active Python environment before API startup. |
| Invalid/incompatible bundle | Confirm provenance and environment versions; rebuild from trusted processed data. Do not bypass validation or load an unknown artifact. |
| CORS failure in browser | Match the exact browser origin in `GLOWGUIDE_CORS_ORIGINS`; `localhost` and `127.0.0.1`, or different ports, are different origins. Restart API. |
| Frontend cannot reach API | Check `/health`, port 8000 and the frontend API URL; restart Next.js after changing its environment. The API must load its bundle successfully first. |
| Port already in use | Stop the process you own, or use API `--port 8001` / frontend `npm run dev -- --port 3001` and update API URL/CORS together. |
| Node install/build failure | Confirm the tested Node/npm versions and run `npm ci` against the committed lockfile; inspect the reported registry or dependency error rather than upgrading packages indiscriminately. |
| Python import failure | Activate the intended environment, install requirements and use `--app-dir src` for uvicorn. If activation is unavailable in PowerShell, invoke `.venv\Scripts\python.exe` directly. |
| Empty recommendations | A valid response may contain no matches. Inspect strategy/filter metadata; relax filters manually. Do not add filler or switch models to hide the result. |
| Unknown profile value | The API accepts normalized strings and backs off to priors; use the frontend's confirmed options for a meaningful profile demonstration. |

## Verification record

Commands executed during this documentation handover, on the verified Windows environment:

| Command | Actual result |
|---|---|
| `python -m pytest -q` | **134 passed, 2 warnings, 84 subtests passed in 8.97s**, exit 0. |
| `python scripts/build_serving_bundle.py` | Success; **1,088,886 interactions, 503,216 users, 2,351 candidates, 2,420 catalog products, 69 cold products**. |
| `npm run lint` | Exit 0, no lint errors/warnings. |
| `npm test` | **2 test files / 21 tests passed**, duration **5.01s**, exit 0. |
| `npm run build` | Next.js **16.3.5**, compilation/TypeScript passed; **3/3 static pages generated**, routes `/` and `/_not-found`, exit 0. |

An in-process FastAPI TestClient smoke check loaded the rebuilt real bundle through the application lifespan: `GET /health` returned HTTP 200 with the counts above; a no-ID Combination/Medium request with `top_k=1` returned HTTP 200, strategy `skin_profile`, and one recommendation. This verifies real artifact loading and request handling, not external HTTP transport or a new browser integration run.

Rebuilt artifact: **37,945,012 bytes (approximately 36.19 MiB)**; build runtime **40.819 seconds**. This differs slightly from the earlier ~36.18 MiB artifact; the serving counts and configuration match. Runtime is environment-dependent and this build ran alongside other checks.

The two backend warnings are upstream deprecations: Starlette's use of httpx in TestClient and the AnyIO `BlockingPortal` alias. No dependencies or application code were changed to suppress them. The earlier small-sample service latency, ~6.015 ms mean / ~12.241 ms P95, was not remeasured here and excludes HTTP transport and bundle loading.

Frozen benchmark JSON and processed-data counts were inspected; hybrid tuning and the four frozen scripts were not rerun for documentation. Preprocessing/install/download commands were checked against their actual source/paths, not reexecuted against unchanged data. Fresh-clone installation, external downloads and Linux execution were not tested in this pass.

## Adding a new model

Implement a separate recommender with the existing `fit`, aligned `product_ids`/`candidate_product_ids`, `score_candidates(*, user_id=...)` and `recommend(seen_product_ids, k, *, user_id=...)` conventions. Add synthetic tests for leakage, determinism, seen exclusion, nonfinite values and explanations. Evaluate in a separately named experiment while preserving frozen split/metric definitions and artifacts.

A new experiment does not automatically enter production. Promotion requires an explicit model-selection decision, serving-bundle lifecycle/compatibility work and routing/API regression tests. Do not modify the frozen benchmark to favor a candidate. No new model is part of this documentation task.

## Operational caveats

There is no authentication, rate limiting, monitoring stack, scheduled retraining, live retailer integration or deployment configuration. State is held in local trusted artifacts and process memory; each application instance loads its own bundle. Treat user IDs and history-derived explanations as sensitive identifiers and behavior. CORS is a browser policy, not access control. See [limitations](LIMITATIONS.md) before extending beyond an authorized local demonstration.
