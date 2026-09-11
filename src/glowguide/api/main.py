"""FastAPI application factory; load one trusted local bundle per lifespan.

Run from the project root:
python -m uvicorn glowguide.api.main:app --app-dir src --reload
GLOWGUIDE_BUNDLE_PATH is trusted operator configuration, never request input.
Relative override paths, like the default, resolve against the project root.
"""

from contextlib import asynccontextmanager
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .schemas import RecommendationRequest, RecommendationResponse, ProductMetadata, HealthResponse
from ..serving import RecommendationService
from ..serving_artifacts import load_serving_bundle


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def create_app(service: RecommendationService | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        if service is not None:
            application.state.recommendation_service = service
        else:
            configured = Path(os.environ.get("GLOWGUIDE_BUNDLE_PATH", "artifacts/serving_bundle.joblib"))
            path = configured if configured.is_absolute() else PROJECT_ROOT / configured
            application.state.recommendation_service = RecommendationService(load_serving_bundle(path))
        try:
            yield
        finally:
            del application.state.recommendation_service

    application = FastAPI(title="GlowGuide Recommendation Service", version="1.0", lifespan=lifespan)
    origins = [origin.strip() for origin in os.environ.get("GLOWGUIDE_CORS_ORIGINS", "http://localhost:3000").split(",") if origin.strip()]
    application.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=False,
                               allow_methods=["GET", "POST"], allow_headers=["Content-Type"])

    @application.exception_handler(RequestValidationError)
    async def invalid_request(request, exc: RequestValidationError):
        # Python's JSON parser accepts nonstandard NaN/Infinity tokens. Pydantic
        # rejects them, but echoing the offending value in a default error can
        # itself fail JSON serialization. Return safe validation details only.
        return JSONResponse(status_code=422, content={"detail": [
            {"loc": list(error["loc"]), "msg": error["msg"], "type": error["type"]}
            for error in exc.errors()
        ]})

    @application.get("/health", response_model=HealthResponse)
    def health():
        return application.state.recommendation_service.health()

    @application.post("/recommendations", response_model=RecommendationResponse)
    def recommendations(request: RecommendationRequest):
        return application.state.recommendation_service.recommend(**request.model_dump())

    @application.get("/products/{product_id}", response_model=ProductMetadata)
    def product(product_id: str):
        metadata = application.state.recommendation_service.product(product_id)
        if metadata is None:
            raise HTTPException(status_code=404, detail="Unknown product")
        return metadata

    return application


app = create_app()
