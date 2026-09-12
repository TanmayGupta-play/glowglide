# Deployment readiness

GlowGuide uses two separately hosted layers: the FastAPI backend in a Linux
container and `frontend/` on a normal Next.js hosting platform, such as Vercel.
This guide prepares and verifies those layers; no cloud deployment has been made.

## A. Backend container

### Prepare the trusted artifacts locally

Run from the repository root. The current trusted bundle records **Python
3.11.8**, and the existing loader compares Python patch and package versions
exactly. The Dockerfile therefore uses the official
`python:3.11.8-slim-bookworm` image, with the unchanged pinned `requirements.txt`.
Use Python 3.11.8 for this bundle build. Do not substitute a floating `3.11-slim`
tag: a different patch version will fail bundle validation. When updating the
runtime for security maintenance, update the image pin and rebuild the trusted
bundle with the same Python and dependency versions, then revalidate the image.
The old patch pin is a compatibility constraint, not a claim of current security
patch coverage.

1. Create a Python environment:

   ```sh
   python --version
   python -m venv .venv
   ```

   Activate in bash with `source .venv/bin/activate`, or in PowerShell with
   `.venv\Scripts\Activate.ps1`.

2. Install dependencies:

   ```sh
   python -m pip install -r requirements.txt
   ```

3. Ensure `data/processed/interactions.csv` and
   `data/processed/products_skincare.csv` exist. If missing, obtain the local raw
   inputs described in [the handover](HANDOVER.md#required-data), then run:

   ```sh
   python scripts/build_processed_data.py
   ```

4. Build the serving bundle locally:

   ```sh
   python scripts/build_serving_bundle.py
   ```

   This uses the existing production refit procedure for Models 0-3. It does not
   rerun hybrid tuning or alter the frozen evaluation methodology.

5. Verify both required files exist and are nonempty:

   ```sh
   python -c "from pathlib import Path; import json; p=Path('artifacts/serving_bundle.joblib'); m=Path('artifacts/serving_manifest.json'); assert p.is_file() and p.stat().st_size > 0; d=json.loads(m.read_text()); assert d['bundle_file_size_bytes'] == p.stat().st_size; print(d['dependency_versions'])"
   ```

   This single-line command also works in PowerShell. Check the reported Python
   version against the Dockerfile and the package versions against
   `requirements.txt`. The inputs are:

   - `artifacts/serving_bundle.joblib`
   - `artifacts/serving_manifest.json`

### Build and run locally

Docker Engine must be running in Linux-container mode. Build from this local
working directory after generating the artifacts, not from a remote Git URL or
an artifact-free checkout:

```sh
docker version
docker build -t glowguide-api .
docker run --rm -p 8000:8000 \
  -e GLOWGUIDE_CORS_ORIGINS=http://localhost:3000 \
  glowguide-api
```

PowerShell equivalent (single line avoids continuation/quoting differences):

```powershell
docker run --rm -p 8000:8000 -e "GLOWGUIDE_CORS_ORIGINS=http://localhost:3000" glowguide-api
```

The service runs as UID/GID 10001, with root-owned application files and
artifacts. It starts one Uvicorn process on `0.0.0.0:8000`, without `--reload`.
The existing application lifespan loads the bundle once; startup never rebuilds
models. From `/app/src/glowguide/api/main.py`, the existing repository-root
resolution points to `/app`, so the default bundle path is already
`/app/artifacts/serving_bundle.joblib`. No path override is needed.

### Health and recommendation checks

After startup completes, request `GET http://localhost:8000/health`:

```sh
curl --fail http://localhost:8000/health
python scripts/verify_deployment.py --api-url http://localhost:8000
```

PowerShell:

```powershell
Invoke-RestMethod -Uri http://localhost:8000/health
python scripts/verify_deployment.py --api-url http://localhost:8000
```

Health must report `status: "ok"` and `bundle_loaded: true`. The smoke checker
uses the existing `httpx` dependency and needs no local bundle or historical
user ID. It posts `{"top_k": 5}` for anonymous popularity and
`{"top_k": 5, "skin_type": "dry", "skin_tone": "light"}` for explicit-profile
recommendations. It checks routing, nonempty results, counts, finite scores,
explanations, bundle-version consistency, and one returned product through
`GET /products/{product_id}`. Failures exit nonzero; success prints a short
summary. It does not print user IDs or response bodies. These are deployment
smoke checks, not model-quality evaluation, a browser CORS test, or a load test.

The Docker health check uses Python's standard library to request
`http://localhost:8000/health` and check both healthy status and bundle loading.
It needs no curl or shell utility. It runs every 30 seconds with a 5-second
process timeout, three retries, and a 120-second startup grace period. Adjust
host startup/readiness probe timing if measured cold starts require more time.
Configure the container platform's HTTP probe to `/health` on port 8000 if it
does not honor Docker health checks. See the
[Docker HEALTHCHECK reference](https://docs.docker.com/reference/dockerfile/#healthcheck).

For a detached validation run, these single-line commands work in bash and
PowerShell. Run the health and script commands once the startup log reports
application startup complete; always stop the test container, including after
a failed check:

```sh
docker build -t glowguide-api:local .
docker run --detach --rm --name glowguide-api-readiness -p 8000:8000 -e "GLOWGUIDE_CORS_ORIGINS=http://localhost:3000" glowguide-api:local
docker logs glowguide-api-readiness
python scripts/verify_deployment.py --api-url http://localhost:8000
docker inspect --format '{{.State.Health.Status}}' glowguide-api-readiness
docker image ls glowguide-api:local
docker stop glowguide-api-readiness
```

`--rm` removes the test container when stopped. The image stays local. Once
validated, the image may be tagged and pushed to your chosen container registry
and run on a compatible container host. Choose the host's CPU architecture when
building if it differs from your local Docker default. Configure HTTPS ingress
to container port 8000, runtime CORS, and `/health` probes there. Registry names,
credentials, and deployment commands depend on your selected account and are
not provided or invented here. No registry push or cloud deployment is part of
this readiness task.

### Build context and artifact safety

`.dockerignore` uses a short allowlist: requirements, `src/` (without Python
caches), the Docker configuration, and exactly the two serving artifacts.
Raw/processed datasets, notebooks, frontend and its `node_modules`, tests,
`.venv`, Git metadata, local environment files, logs including `eda_output.txt`,
and other artifacts never enter the build context. Explicit `COPY` instructions
package only requirements, source, and the bundle/manifest. Missing either
artifact causes Docker's required `COPY` to fail with a missing-source/checksum
error; generate the artifacts locally and retry. There is no download or model
build fallback. This missing-file failure has only been checked statically
until a real Docker build is run.

Git ignore rules and Docker context rules are separate; preserving the two
files in Docker's context does not make them tracked by Git. Both serving files
remain ignored under the existing `artifacts/*` Git policy. Never force-add
binary serving artifacts. See [Docker build contexts and ignore rules](https://docs.docker.com/build/concepts/context/#dockerignore-files).

**Load joblib/pickle-style artifacts only from trusted sources.** Deserialization
can execute code; validation is not a sandbox. Build locally from trusted
project data, never download arbitrary joblib files or accept uploaded bundles.
The image contains historical serving state, so control image/registry access
accordingly even though raw datasets are excluded.

## B. Frontend hosting

1. Select `frontend/` as the Next.js project root on the hosting platform. The
   existing `build` (`next build`) and `start` (`next start`) scripts support
   ordinary Next.js hosting. No `vercel.json` or frontend Dockerfile is required.
   The local toolchain is Node.js 22.19.0 / npm 10.9.3; use a compatible supported
   Node runtime on the chosen host.
2. Install from the lockfile with `npm ci`. Set this variable in the platform's
   **build environment** before building:

   ```dotenv
   NEXT_PUBLIC_GLOWGUIDE_API_URL=https://your-backend-domain.example
   ```

   Replace the placeholder with the browser-accessible HTTPS backend base URL.
   `frontend/.env.production.example` documents the setting; example files are
   not automatically loaded by Next.js. For local production builds, copy it to
   the ignored `frontend/.env.production.local` and edit the placeholder, or set
   the environment variable directly. `frontend/.env.example` retains localhost
   for development. `frontend/lib/api.ts` reads only
   `NEXT_PUBLIC_GLOWGUIDE_API_URL`, falling back to `http://localhost:8000` if it
   is unset; do not leave it unset in a hosted production build.
3. Run from `frontend/`:

   ```sh
   npm run lint
   npm test
   npm run build
   ```

   For a self-managed Node host, start the built app with `npm run start`.
4. Set the backend's runtime CORS allowlist to the final frontend origin:

   ```dotenv
   GLOWGUIDE_CORS_ORIGINS=https://your-frontend-domain.example
   ```

   Multiple exact origins can be comma-separated. Use scheme and hostname (plus
   port when needed), without paths or trailing slashes. Development remains
   `http://localhost:3000`. Do not configure wildcard CORS with credentials;
   the existing backend has `allow_credentials=False`. Restart/recreate the
   backend when changing its CORS environment. Check recommendation and product
   detail requests in the hosted browser to confirm HTTPS connectivity and CORS.

`NEXT_PUBLIC_*` values are embedded in browser JavaScript at **build time**.
Changing `NEXT_PUBLIC_GLOWGUIDE_API_URL` requires a frontend rebuild and a new
frontend deployment; changing only a running server's environment is not
enough. This behavior was checked against the installed Next.js environment
variable guide in `frontend/node_modules/next/dist/docs/`.

Compose is intentionally omitted: the frontend already targets separate Next.js
hosting and has no production container setup. For a local demo, run the backend
container and `npm run build` / `npm run start` in `frontend/`.

## Environment variables and secrets

Inference currently requires no application secrets. Do not add fake JWT or API
keys. Hosting/registry credentials, if later needed, belong in the chosen
platform's credential management, never in committed examples.

| Variable | Where / when | Purpose / default |
| --- | --- | --- |
| `GLOWGUIDE_BUNDLE_PATH` | Backend startup; optional | Trusted operator override; defaults to `artifacts/serving_bundle.joblib`, resolved against `/app` in the image. Absolute container paths are supported. No Windows machine path is needed. |
| `GLOWGUIDE_CORS_ORIGINS` | Backend startup; set for production | Comma-separated exact browser origins; defaults to `http://localhost:3000`. Set the final frontend origin. |
| `NEXT_PUBLIC_GLOWGUIDE_API_URL` | Frontend build; set for production | Public browser API base URL; defaults to `http://localhost:8000`. Not a secret; changes require rebuilding. |

## Production limitations

- No authentication or rate limiting. CORS is not access control for API callers.
- No persistent database; inference uses the packaged serving state.
- The serving bundle is immutable to the service user inside the image. Model
  refresh requires rebuilding the bundle locally, rebuilding the image, and
  replacing running containers. No startup training or automatic refresh exists.
- No monitoring/observability stack yet; container logs and basic health are
  available, but do not provide full production monitoring.
- The container host must provide enough memory for all loaded models and
  serving state. No exact cloud memory requirement has been measured. More
  workers/replicas each load their own bundle and increase memory usage.
- Cold starts include model deserialization and validation. Measure startup and
  steady-state memory on the selected host before choosing its resource limits.
- The unchanged full `requirements.txt` includes development/analysis packages;
  pip's cache is disabled, but this is not a minimal inference-only dependency
  set. Image size and Linux dependency installation still need a real build.

## Readiness verification record

Record from this Windows workspace on 2026-09-12:

- Docker CLI 29.2.1 is installed. `docker version` could not connect to
  `dockerDesktopLinuxEngine`, including outside the workspace sandbox: the
  daemon's named pipe was absent. A usable Docker Engine was unavailable.
- Actual image build, image size, container health, and container smoke-check
  results are **not measured**. Start Docker Desktop/Engine and run the detached
  validation sequence above before treating the image as runtime-validated.
- Dockerfile sources, repository-relative bundle resolution, exact Python and
  pinned dependency compatibility, and Git artifact ignore rules were inspected
  statically. The trusted bundle and manifest were already present and were not
  rebuilt by this readiness task.
- `python -m pytest -q`: **146 passed, 84 subtests passed, 2 warnings**, 6.71s.
  This includes 12 deployment-check tests covering success, bad health/routing,
  empty or inconsistent results, product mismatch, HTTP/JSON failures, and a
  nonzero CLI failure. Warnings are existing upstream Starlette/httpx and AnyIO
  deprecations.
- Frontend `npm run lint`: passed with no errors/warnings. `npm test`: **21 tests
  passed in 2 files**, 18.94s. `npm run build`: passed compilation, TypeScript,
  and prerendering of `/` and `/_not-found`.
- A temporary local Windows Uvicorn process loaded the existing trusted bundle
  through the unchanged default path. `/health` returned HTTP 200 with
  `bundle_loaded=true`. `python scripts/verify_deployment.py --api-url
  http://localhost:8000` passed: 5 popularity recommendations, 5 skin-profile
  recommendations, and a matching product lookup. The localhost frontend CORS
  preflight also passed. The temporary API was stopped afterward. These results
  validate the local API and checker, **not a Linux container**.
- SHA-256 checks confirmed the serving bundle, manifest, and pre-existing
  `eda_output.txt` were unchanged. No source ML or evaluation code was modified;
  no tuning was rerun, and nothing was staged, committed, pushed, or deployed.
