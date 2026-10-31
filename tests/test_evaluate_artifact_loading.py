from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch

from recsys.utils.artifacts import sha256_file
from scripts.evaluate import load_and_validate_two_tower_item_embeddings


def test_evaluation_loads_item_embeddings_from_pinned_release_path(
    tmp_path: Path,
) -> None:
    item_embeddings_path = tmp_path / "item_embeddings.pt"
    embeddings = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        dtype=torch.float32,
    )
    torch.save(embeddings, item_embeddings_path)

    expected_lineage = {
        "data_run_id": "model-data-001",
        "dataset_fingerprint": "a" * 64,
    }
    checkpoint_sha256 = "b" * 64

    metadata_path = tmp_path / "embedding_metadata.json"
    metadata = {
        "metadata_version": 1,
        "model": "TwoTowerRecommender",
        "parent_checkpoint": {"sha256": checkpoint_sha256},
        "lineage": expected_lineage,
        "item_embeddings": {
            "sha256": sha256_file(item_embeddings_path),
            "shape": [3, 2],
        },
    }
    metadata_path.write_text(
        json.dumps(metadata, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    model_release = SimpleNamespace(
        item_embeddings_path=item_embeddings_path,
        embedding_metadata_path=metadata_path,
    )
    checkpoint = {"model_config": {"embedding_dim": 2}}

    loaded, loaded_sha256 = load_and_validate_two_tower_item_embeddings(
        device=torch.device("cpu"),
        model_release=model_release,
        two_tower_checkpoint=checkpoint,
        two_tower_checkpoint_sha256=checkpoint_sha256,
        expected_lineage=expected_lineage,
        n_items=3,
    )

    assert torch.equal(loaded, embeddings)
    assert loaded_sha256 == sha256_file(item_embeddings_path)
