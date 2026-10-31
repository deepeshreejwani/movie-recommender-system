from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest
import torch

import scripts.train as train_script
from recsys.config import COLD_ITEM_INDEX
from recsys.models.two_tower import TwoTowerRecommender
from recsys.utils.artifacts import sha256_file
from scripts.train import (
    build_training_arrays,
    export_two_tower_embeddings,
    prepare_final_training_interactions,
)


def test_two_tower_embedding_export_on_cpu() -> None:
    """
    Final Two-Tower embedding export must work on CPU and
    produce one embedding for every user and item.
    """

    torch.manual_seed(42)

    device = torch.device("cpu")

    n_users = 5
    n_items = 7
    embedding_dim = 4

    model = TwoTowerRecommender(
        n_users=n_users,
        n_items=n_items,
        embedding_dim=embedding_dim,
        hidden_dim=8,
    ).to(device)

    (
        user_embeddings,
        item_embeddings,
    ) = export_two_tower_embeddings(
        model=model,
        device=device,
        n_users=n_users,
        n_items=n_items,
    )

    assert user_embeddings.shape == (
        n_users,
        embedding_dim,
    )

    assert item_embeddings.shape == (
        n_items,
        embedding_dim,
    )

    assert user_embeddings.device.type == "cpu"
    assert item_embeddings.device.type == "cpu"

    assert user_embeddings.requires_grad is False
    assert item_embeddings.requires_grad is False

    expected_users = model.get_user_embeddings(
        device=device,
    )

    expected_items = model.get_item_embeddings(
        device=device,
    )

    assert torch.equal(
        user_embeddings,
        expected_users,
    )

    assert torch.equal(
        item_embeddings,
        expected_items,
    )


def test_two_tower_embedding_export_rejects_catalog_mismatch() -> None:
    """
    Export must fail if the expected vocabulary size does not
    match the model's actual embedding matrices.
    """

    device = torch.device("cpu")

    model = TwoTowerRecommender(
        n_users=5,
        n_items=7,
        embedding_dim=4,
        hidden_dim=8,
    ).to(device)

    with pytest.raises(
        RuntimeError,
        match="user embedding count",
    ):
        export_two_tower_embeddings(
            model=model,
            device=device,
            n_users=6,
            n_items=7,
        )

    with pytest.raises(
        RuntimeError,
        match="item embedding count",
    ):
        export_two_tower_embeddings(
            model=model,
            device=device,
            n_users=5,
            n_items=8,
        )


def test_final_training_uses_only_warm_validation_items() -> None:
    """
    Final production training must use:

        train + warm validation

    Cold validation targets encoded as -1 must be excluded.
    """

    train = pl.DataFrame(
        {
            "user_idx": [
                0,
                1,
                2,
            ],
            "item_idx": [
                0,
                1,
                2,
            ],
        }
    )

    validation = pl.DataFrame(
        {
            "user_idx": [
                0,
                1,
                2,
            ],
            "item_idx": [
                2,
                COLD_ITEM_INDEX,
                0,
            ],
        }
    )

    (
        interactions,
        warm_validation_count,
        cold_validation_count,
    ) = prepare_final_training_interactions(
        train=train,
        validation=validation,
        n_users=3,
        n_items=3,
    )

    assert interactions.height == 5

    assert warm_validation_count == 2
    assert cold_validation_count == 1

    assert interactions.filter(pl.col("item_idx") < 0).is_empty()


def test_training_arrays_reject_cold_item_index() -> None:
    """
    item_idx=-1 must never reach embedding-model training.
    """

    interactions = pl.DataFrame(
        {
            "user_idx": [
                0,
                1,
            ],
            "item_idx": [
                0,
                COLD_ITEM_INDEX,
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="negative item indices",
    ):
        build_training_arrays(
            interactions=interactions,
            n_users=2,
            n_items=2,
        )


def test_model_release_retry_repoints_existing_identical_release(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    models_dir = tmp_path / "models"
    releases_dir = models_dir / "releases"
    pointer_path = models_dir / "current_release.json"

    monkeypatch.setattr(train_script, "MODEL_RELEASES_DIR", releases_dir)
    monkeypatch.setattr(train_script, "MODEL_RELEASE_POINTER_PATH", pointer_path)

    model_release_id = "model-retry"
    final_release_dir = releases_dir / model_release_id
    final_release_dir.mkdir(parents=True)

    manifest_path = final_release_dir / train_script.MODEL_RELEASE_MANIFEST_NAME
    manifest_path.write_text('{"same": true}\n', encoding="utf-8")
    manifest_sha256 = sha256_file(manifest_path)

    staging_dir = releases_dir / ".staging-retry"
    staging_dir.mkdir(parents=True)
    (staging_dir / train_script.MODEL_RELEASE_MANIFEST_NAME).write_text(
        '{"same": true}\n',
        encoding="utf-8",
    )

    lineage = {
        "release_lineage_id": "release-123",
        "release_fingerprint": "a" * 64,
        "data_run_id": "model-data-123",
        "dataset_fingerprint": "b" * 64,
    }

    published = train_script.publish_staged_model_release(
        staging_dir=staging_dir,
        model_release_id=model_release_id,
        manifest_sha256=manifest_sha256,
        lineage=lineage,
    )

    assert published == final_release_dir
    assert not staging_dir.exists()

    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    assert pointer["model_release_id"] == model_release_id
    assert pointer["manifest_sha256"] == manifest_sha256
