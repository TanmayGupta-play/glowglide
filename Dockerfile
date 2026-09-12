# Match the exact Python version recorded in the trusted serving bundle.
# When upgrading Python, rebuild the bundle locally with the same version.
FROM python:3.11.8-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
# Required local inputs: run python scripts/build_serving_bundle.py first.
# Explicit COPY instructions fail the build if either artifact is missing.
COPY artifacts/serving_bundle.joblib ./artifacts/serving_bundle.joblib
COPY artifacts/serving_manifest.json ./artifacts/serving_manifest.json

# Root-owned application and artifacts remain read-only to the service user.
USER 10001:10001

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD ["python", "-c", "import json, sys, urllib.request; r = urllib.request.urlopen('http://localhost:8000/health', timeout=3); h = json.load(r); sys.exit(0 if r.status == 200 and h.get('status') == 'ok' and h.get('bundle_loaded') is True else 1)"]

CMD ["python", "-m", "uvicorn", "glowguide.api.main:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]
