from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import (
    FastAPI,
    HTTPException,
)
from pydantic import (
    BaseModel,
    Field,
)

from recsys.config import (
    API_DEFAULT_K,
    API_MAX_K,
)
from recsys.serving.inference import (
    RecommendationService,
)

logger = logging.getLogger(__name__)

service = RecommendationService()

PositiveUserId = Annotated[
    int,
    Field(gt=0),
]


@asynccontextmanager
async def lifespan(
    app: FastAPI,
) -> AsyncIterator[None]:
    """
    Attempt to load the recommendation service at startup.

    A serving-artifact failure must not terminate the API
    process. The application remains available so readiness
    can be reported through /health.

    The original startup error is retained internally on
    app.state for diagnostics.
    """

    app.state.startup_error = None

    try:
        service.load()

    except Exception as exc:
        logger.exception("Recommendation service failed to load during startup.")

        service.loaded = False
        service.manifest_validated = False

        app.state.startup_error = str(exc)

    yield


app = FastAPI(
    title="Movie Recommender System",
    description=(
        "Two-stage movie recommendation API using "
        "Two-Tower retrieval and NeuMF ranking."
    ),
    version="1.1.0",
    lifespan=lifespan,
)


class RecommendationRequest(
    BaseModel,
):
    user_id: PositiveUserId

    k: int = Field(
        default=API_DEFAULT_K,
        ge=1,
        le=API_MAX_K,
    )


class BatchRecommendationRequest(
    BaseModel,
):
    user_ids: list[PositiveUserId] = Field(
        ...,
        min_length=1,
        max_length=100,
    )

    k: int = Field(
        default=API_DEFAULT_K,
        ge=1,
        le=API_MAX_K,
    )


class MovieRecommendation(
    BaseModel,
):
    movie_id: int
    title: str
    genres: str
    score: float


class RecommendationResponse(
    BaseModel,
):
    user_id: int
    k: int
    latency_ms: float

    recommendations: list[MovieRecommendation]


class BatchRecommendationResult(
    BaseModel,
):
    user_id: int
    k: int

    recommendations: list[MovieRecommendation]


class BatchRecommendationResponse(
    BaseModel,
):
    batch_size: int

    batch_latency_ms: float

    average_latency_ms_per_user: float

    results: list[BatchRecommendationResult]


class HealthResponse(
    BaseModel,
):
    status: str
    model_loaded: bool
    manifest_validated: bool


def ensure_service_ready() -> None:
    """
    Require a fully loaded and manifest-validated serving bundle.

    The API process may be alive while recommendation artifacts
    are unavailable. Such a process is not ready for inference.
    """

    if not service.loaded:
        raise HTTPException(
            status_code=503,
            detail="Recommendation service is not loaded.",
        )

    if not service.manifest_validated:
        raise HTTPException(
            status_code=503,
            detail="Recommendation serving bundle is not validated.",
        )


@app.get(
    "/health",
    response_model=HealthResponse,
)
def health() -> HealthResponse:
    """
    Report service readiness.

    HTTP 200 means the serving bundle is loaded, validated,
    and ready for inference.

    HTTP 503 means the API process is alive but the
    recommendation service is not ready.
    """

    ensure_service_ready()

    return HealthResponse(
        status="healthy",
        model_loaded=True,
        manifest_validated=True,
    )


@app.get(
    "/model/info",
)
def model_info() -> dict[
    str,
    object,
]:
    """Return loaded model and serving-bundle information."""

    ensure_service_ready()

    try:
        return service.model_info()

    except RuntimeError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc


@app.post(
    "/recommend",
    response_model=RecommendationResponse,
)
def recommend(
    request: RecommendationRequest,
) -> RecommendationResponse:
    """Generate recommendations for one known user."""

    ensure_service_ready()

    try:
        (
            recommendations,
            latency_ms,
        ) = service.recommend(
            user_id=request.user_id,
            k=request.k,
        )

    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except RuntimeError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc

    return RecommendationResponse(
        user_id=request.user_id,
        k=request.k,
        latency_ms=latency_ms,
        recommendations=[
            MovieRecommendation(
                movie_id=item.movie_id,
                title=item.title,
                genres=item.genres,
                score=item.score,
            )
            for item in recommendations
        ],
    )


@app.post(
    "/recommend/batch",
    response_model=BatchRecommendationResponse,
)
def recommend_batch(
    request: BatchRecommendationRequest,
) -> BatchRecommendationResponse:
    """
    Generate recommendations for multiple users
    in one batched inference pass.

    `batch_latency_ms` is the measured end-to-end
    latency for the complete batch.

    `average_latency_ms_per_user` is derived from
    batch latency and is not an independently
    measured per-user latency.
    """

    ensure_service_ready()

    try:
        (
            recommendations_batch,
            batch_latency_ms,
        ) = service.recommend_batch(
            user_ids=request.user_ids,
            k=request.k,
        )

    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except RuntimeError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc

    if len(recommendations_batch) != len(request.user_ids):
        raise HTTPException(
            status_code=500,
            detail=("Recommendation batch size does not match request size."),
        )

    batch_size = len(request.user_ids)

    average_latency_ms = batch_latency_ms / batch_size

    results = [
        BatchRecommendationResult(
            user_id=user_id,
            k=request.k,
            recommendations=[
                MovieRecommendation(
                    movie_id=item.movie_id,
                    title=item.title,
                    genres=item.genres,
                    score=item.score,
                )
                for item in recommendations
            ],
        )
        for (
            user_id,
            recommendations,
        ) in zip(
            request.user_ids,
            recommendations_batch,
            strict=True,
        )
    ]

    return BatchRecommendationResponse(
        batch_size=batch_size,
        batch_latency_ms=batch_latency_ms,
        average_latency_ms_per_user=round(
            average_latency_ms,
            3,
        ),
        results=results,
    )
