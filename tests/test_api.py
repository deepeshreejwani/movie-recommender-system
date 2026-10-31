from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import app.main as api
from recsys.serving.inference import Recommendation


@pytest.fixture
def client(
    monkeypatch: pytest.MonkeyPatch,
):
    def fake_load() -> None:
        api.service.loaded = True
        api.service.manifest_validated = True

    def fake_model_info() -> dict[str, object]:
        return {
            "loaded": True,
            "manifest_validated": True,
            "serving_bundle_version": 2,
            "device": "cpu",
            "users": 163_760,
            "items": 17_818,
            "candidate_size": 500,
            "max_k": 50,
            "retrieval_model": "TwoTowerRecommender",
            "ranking_model": "NeuMF",
        }

    def fake_recommend(
        user_id: int,
        k: int,
    ) -> tuple[
        list[Recommendation],
        float,
    ]:
        if user_id == 999_999:
            raise KeyError(f"Unknown user_id: {user_id}")

        if user_id == 888_888:
            raise RuntimeError("Serving artifacts are incomplete.")

        recommendations = [
            Recommendation(
                movie_id=100 + index,
                title=f"Movie {index}",
                genres="Drama",
                score=round(
                    5.0 - index * 0.1,
                    3,
                ),
            )
            for index in range(k)
        ]

        return (
            recommendations,
            2.345,
        )

    def fake_recommend_batch(
        user_ids: list[int],
        k: int,
    ) -> tuple[
        list[list[Recommendation]],
        float,
    ]:
        results: list[list[Recommendation]] = []

        for user_id in user_ids:
            if user_id == 999_999:
                raise KeyError(f"Unknown user_id: {user_id}")

            if user_id == 888_888:
                raise RuntimeError("Serving artifacts are incomplete.")

            recommendations = [
                Recommendation(
                    movie_id=100 + index,
                    title=f"Movie {index}",
                    genres="Drama",
                    score=round(
                        5.0 - index * 0.1,
                        3,
                    ),
                )
                for index in range(k)
            ]

            results.append(recommendations)

        return (
            results,
            8.0,
        )

    monkeypatch.setattr(
        api.service,
        "load",
        fake_load,
    )

    monkeypatch.setattr(
        api.service,
        "model_info",
        fake_model_info,
    )

    monkeypatch.setattr(
        api.service,
        "recommend",
        fake_recommend,
    )

    monkeypatch.setattr(
        api.service,
        "recommend_batch",
        fake_recommend_batch,
    )

    with TestClient(api.app) as test_client:
        yield test_client

    api.service.loaded = False
    api.service.manifest_validated = False


def test_health(
    client: TestClient,
) -> None:
    response = client.get("/health")

    assert response.status_code == 200

    payload = response.json()

    assert payload == {
        "status": "healthy",
        "model_loaded": True,
        "manifest_validated": True,
    }


def test_health_returns_503_when_unloaded(
    client: TestClient,
) -> None:
    api.service.loaded = False

    response = client.get("/health")

    assert response.status_code == 503

    assert "not loaded" in response.json()["detail"]

    api.service.loaded = True


def test_health_returns_503_when_manifest_is_not_validated(
    client: TestClient,
) -> None:
    api.service.manifest_validated = False

    response = client.get("/health")

    assert response.status_code == 503

    assert "not validated" in response.json()["detail"]

    api.service.manifest_validated = True


def test_startup_load_failure_keeps_api_alive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    A serving-artifact failure during application startup
    must not terminate the API process.

    The API remains reachable and reports HTTP 503 readiness.
    """

    startup_message = "Serving manifest is missing."

    def failing_load() -> None:
        raise RuntimeError(startup_message)

    api.service.loaded = False
    api.service.manifest_validated = False

    monkeypatch.setattr(
        api.service,
        "load",
        failing_load,
    )

    with TestClient(api.app) as failed_client:
        health_response = failed_client.get("/health")

        assert health_response.status_code == 503

        assert "not loaded" in health_response.json()["detail"]

        # Startup exception is retained for internal
        # diagnostics rather than crashing the process.
        assert api.app.state.startup_error == startup_message

        assert api.service.loaded is False

        assert api.service.manifest_validated is False

        # Other model-dependent routes must also remain
        # unavailable while the process itself stays alive.
        model_info_response = failed_client.get("/model/info")

        assert model_info_response.status_code == 503


def test_model_info(
    client: TestClient,
) -> None:
    response = client.get("/model/info")

    assert response.status_code == 200

    payload = response.json()

    assert payload["loaded"] is True

    assert payload["manifest_validated"] is True

    assert payload["serving_bundle_version"] == 2

    assert payload["users"] == 163_760

    assert payload["items"] == 17_818

    assert payload["candidate_size"] == 500

    assert payload["max_k"] == 50

    assert payload["retrieval_model"] == "TwoTowerRecommender"

    assert payload["ranking_model"] == "NeuMF"


def test_recommend(
    client: TestClient,
) -> None:
    response = client.post(
        "/recommend",
        json={
            "user_id": 1,
            "k": 5,
        },
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["user_id"] == 1

    assert payload["k"] == 5

    assert payload["latency_ms"] == 2.345

    assert len(payload["recommendations"]) == 5

    first_item = payload["recommendations"][0]

    assert first_item["movie_id"] == 100

    assert first_item["title"] == "Movie 0"

    assert first_item["genres"] == "Drama"

    assert first_item["score"] == 5.0


def test_unknown_user(
    client: TestClient,
) -> None:
    response = client.post(
        "/recommend",
        json={
            "user_id": 999_999,
            "k": 10,
        },
    )

    assert response.status_code == 404

    assert "Unknown user_id" in response.json()["detail"]


def test_runtime_serving_failure_returns_503(
    client: TestClient,
) -> None:
    response = client.post(
        "/recommend",
        json={
            "user_id": 888_888,
            "k": 10,
        },
    )

    assert response.status_code == 503

    assert "incomplete" in response.json()["detail"]


def test_invalid_k(
    client: TestClient,
) -> None:
    response = client.post(
        "/recommend",
        json={
            "user_id": 1,
            "k": 100,
        },
    )

    assert response.status_code == 422


def test_invalid_user_id(
    client: TestClient,
) -> None:
    response = client.post(
        "/recommend",
        json={
            "user_id": 0,
            "k": 10,
        },
    )

    assert response.status_code == 422


def test_batch_recommend(
    client: TestClient,
) -> None:
    response = client.post(
        "/recommend/batch",
        json={
            "user_ids": [
                1,
                2,
                3,
                4,
            ],
            "k": 5,
        },
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["batch_size"] == 4

    assert payload["batch_latency_ms"] == 8.0

    assert payload["average_latency_ms_per_user"] == 2.0

    results = payload["results"]

    assert len(results) == 4

    for (
        expected_user_id,
        result,
    ) in zip(
        [
            1,
            2,
            3,
            4,
        ],
        results,
        strict=True,
    ):
        assert result["user_id"] == expected_user_id

        assert result["k"] == 5

        assert "latency_ms" not in result

        assert len(result["recommendations"]) == 5


def test_batch_unknown_user(
    client: TestClient,
) -> None:
    response = client.post(
        "/recommend/batch",
        json={
            "user_ids": [
                1,
                999_999,
            ],
            "k": 5,
        },
    )

    assert response.status_code == 404

    assert "Unknown user_id" in response.json()["detail"]


def test_batch_runtime_failure_returns_503(
    client: TestClient,
) -> None:
    response = client.post(
        "/recommend/batch",
        json={
            "user_ids": [
                1,
                888_888,
            ],
            "k": 5,
        },
    )

    assert response.status_code == 503


def test_batch_size_validation(
    client: TestClient,
) -> None:
    response = client.post(
        "/recommend/batch",
        json={
            "user_ids": [],
            "k": 10,
        },
    )

    assert response.status_code == 422


def test_batch_user_id_validation(
    client: TestClient,
) -> None:
    response = client.post(
        "/recommend/batch",
        json={
            "user_ids": [
                1,
                0,
                3,
            ],
            "k": 10,
        },
    )

    assert response.status_code == 422


def test_batch_max_size_validation(
    client: TestClient,
) -> None:
    response = client.post(
        "/recommend/batch",
        json={
            "user_ids": list(
                range(
                    1,
                    102,
                )
            ),
            "k": 10,
        },
    )

    assert response.status_code == 422
