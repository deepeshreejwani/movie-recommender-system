from __future__ import annotations

import numpy as np
import pytest

from recsys.config import (
    RANKING_CANDIDATE_SIZE,
    SERVING_DIR,
)
from recsys.serving.inference import (
    RecommendationService,
    resolve_current_serving_release,
)

SERVING_POINTER_PATH = SERVING_DIR / "current_release.json"


def test_serving_release_to_real_recommendation_e2e() -> None:
    """
    Real serving integration test.

    Exercises:
        serving pointer
        -> immutable serving release
        -> pinned immutable model release
        -> RecommendationService.load()
        -> real recommendation
        -> serving invariants

    This test intentionally uses already-built production artifacts.
    It does not train models or rebuild serving artifacts.
    """

    if not SERVING_POINTER_PATH.is_file():
        pytest.skip(
            "Serving artifacts are unavailable. "
            "Run scripts/build_serving_artifacts.py first."
        )

    # ---------------------------------------------------------
    # Resolve immutable serving release
    # ---------------------------------------------------------

    serving_release = resolve_current_serving_release()

    assert serving_release.release_dir.is_dir()
    assert serving_release.manifest_path.is_file()

    assert serving_release.serving_release_id == serving_release.release_dir.name

    # ---------------------------------------------------------
    # Load complete serving system
    # ---------------------------------------------------------

    service = RecommendationService()

    service.load()

    assert service.loaded is True
    assert service.manifest_validated is True

    assert service.serving_release is not None
    assert service.model_release is not None
    assert service.manifest is not None
    assert service.lineage is not None

    # The service must load the exact serving generation
    # selected by current_release.json.
    assert (
        service.serving_release.serving_release_id == serving_release.serving_release_id
    )

    assert (
        service.serving_release.serving_release_fingerprint
        == serving_release.serving_release_fingerprint
    )

    # ---------------------------------------------------------
    # Verify serving -> model-release pinning
    # ---------------------------------------------------------

    release_metadata = service.manifest["release"]

    assert (
        service.model_release.model_release_id == release_metadata["model_release_id"]
    )

    assert (
        service.model_release.model_release_fingerprint
        == release_metadata["model_release_fingerprint"]
    )

    assert service.model_release.lineage == service.lineage

    # ---------------------------------------------------------
    # Catalog / runtime invariants
    # ---------------------------------------------------------

    assert service.n_users > 0
    assert service.n_items > 0

    assert len(service.user_id_to_idx) == service.n_users

    assert service.movie_ids is not None
    assert service.movie_titles is not None
    assert service.movie_genres is not None
    assert service.item_embeddings is not None
    assert service.seen_items is not None

    assert len(service.movie_ids) == service.n_items
    assert len(service.movie_titles) == service.n_items
    assert len(service.movie_genres) == service.n_items

    assert service.item_embeddings.shape[0] == service.n_items

    # ---------------------------------------------------------
    # Real recommendation
    # ---------------------------------------------------------

    user_id = min(service.user_id_to_idx)

    recommendations, latency_ms = service.recommend(
        user_id=user_id,
        k=10,
    )

    assert len(recommendations) == 10
    assert latency_ms >= 0.0

    recommended_movie_ids = [
        recommendation.movie_id for recommendation in recommendations
    ]

    assert len(set(recommended_movie_ids)) == 10

    for recommendation in recommendations:
        assert isinstance(recommendation.movie_id, int)
        assert recommendation.title
        assert recommendation.genres
        assert np.isfinite(recommendation.score)

    # ---------------------------------------------------------
    # Recommended items must belong to the frozen catalog
    # ---------------------------------------------------------

    movie_id_to_idx = {
        int(movie_id): item_idx for item_idx, movie_id in enumerate(service.movie_ids)
    }

    recommended_item_indices = [
        movie_id_to_idx[movie_id] for movie_id in recommended_movie_ids
    ]

    assert all(0 <= item_idx < service.n_items for item_idx in recommended_item_indices)

    # ---------------------------------------------------------
    # Seen items must not be recommended
    # ---------------------------------------------------------

    user_idx = service.user_id_to_idx[user_id]

    seen_start = service.seen_items.indptr[user_idx]
    seen_end = service.seen_items.indptr[user_idx + 1]

    seen_item_indices = set(service.seen_items.indices[seen_start:seen_end].tolist())

    assert not (seen_item_indices & set(recommended_item_indices))

    # ---------------------------------------------------------
    # Model information contract
    # ---------------------------------------------------------

    model_info = service.model_info()

    assert model_info["loaded"] is True
    assert model_info["manifest_validated"] is True

    assert model_info["serving_release_id"] == serving_release.serving_release_id

    assert model_info["model_release_id"] == service.model_release.model_release_id

    assert model_info["users"] == service.n_users
    assert model_info["items"] == service.n_items

    assert model_info["candidate_size"] == RANKING_CANDIDATE_SIZE

    assert model_info["retrieval_model"] == "TwoTowerRecommender"
    assert model_info["ranking_model"] == "NeuMF"
