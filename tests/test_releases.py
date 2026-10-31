from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from recsys.utils import releases
from recsys.utils.artifacts import (
    canonical_json_sha256,
    sha256_file,
)


def _write_json(
    path: Path,
    payload: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            payload,
            indent=4,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _artifact_metadata(
    path: Path,
) -> dict[str, Any]:
    return {
        "filename": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _configure_release_paths(
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
) -> None:
    releases_dir = root / "releases"

    monkeypatch.setattr(
        releases,
        "FINAL_NEUMF_MODEL_PATH",
        root / "final_neumf.pt",
    )

    monkeypatch.setattr(
        releases,
        "MODEL_RELEASES_DIR",
        releases_dir,
    )

    monkeypatch.setattr(
        releases,
        "MODEL_RELEASE_POINTER_PATH",
        root / "current_release.json",
    )


def _build_valid_release(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> tuple[
    releases.ModelRelease,
    dict[str, Any],
    Path,
]:
    root = tmp_path / "models"

    _configure_release_paths(
        monkeypatch=monkeypatch,
        root=root,
    )

    lineage = {
        "data_run_id": "data-run-001",
        "dataset_fingerprint": "1" * 64,
        "preprocessing_protocol_version": "preprocessing-v1",
        "final_training_protocol": "train+warm_validation",
        "train_sha256": "2" * 64,
        "warm_validation_sha256": "3" * 64,
        "user_mapping_sha256": "4" * 64,
        "item_mapping_sha256": "5" * 64,
        "release_lineage_id": "release-lineage-001",
        "release_fingerprint": "6" * 64,
    }

    temporary_release_dir = releases.MODEL_RELEASES_DIR / "temporary-release"

    temporary_release_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    artifact_contents = {
        "neumf_model": (
            "final_neumf.pt",
            b"neumf-checkpoint",
        ),
        "two_tower_model": (
            "final_two_tower.pt",
            b"two-tower-checkpoint",
        ),
        "two_tower_user_embeddings": (
            "final_two_tower_user_embeddings.pt",
            b"user-embeddings",
        ),
        "two_tower_item_embeddings": (
            "final_two_tower_item_embeddings.pt",
            b"item-embeddings",
        ),
        "two_tower_embedding_metadata": (
            "final_two_tower_embeddings_metadata.json",
            b'{"metadata_version":1}\n',
        ),
    }

    artifact_metadata: dict[str, dict[str, Any]] = {}

    for artifact_name, (
        filename,
        contents,
    ) in artifact_contents.items():
        artifact_path = temporary_release_dir / filename

        artifact_path.write_bytes(contents)

        artifact_metadata[artifact_name] = _artifact_metadata(artifact_path)

    artifact_hashes = {
        artifact_name: str(metadata["sha256"])
        for artifact_name, metadata in sorted(artifact_metadata.items())
    }

    release_identity = {
        "model_release_manifest_version": (releases.MODEL_RELEASE_MANIFEST_VERSION),
        "lineage": dict(lineage),
        "artifact_hashes": (artifact_hashes),
    }

    model_release_fingerprint = canonical_json_sha256(release_identity)

    model_release_id = f"model-{model_release_fingerprint[:20]}"

    release_dir = releases.MODEL_RELEASES_DIR / model_release_id

    temporary_release_dir.rename(release_dir)

    manifest = {
        "model_release_manifest_version": (releases.MODEL_RELEASE_MANIFEST_VERSION),
        "model_release_id": (model_release_id),
        "model_release_fingerprint": (model_release_fingerprint),
        "lineage": dict(lineage),
        "catalog": {
            "n_users": 3,
            "n_items": 5,
        },
        "artifacts": (artifact_metadata),
    }

    manifest_path = release_dir / releases.MODEL_RELEASE_MANIFEST_NAME

    _write_json(
        manifest_path,
        manifest,
    )

    pointer = {
        "pointer_version": (releases.MODEL_RELEASE_POINTER_VERSION),
        "model_release_id": (model_release_id),
        "model_release_dir": str(Path("releases") / model_release_id),
        "manifest_filename": (releases.MODEL_RELEASE_MANIFEST_NAME),
        "manifest_sha256": (sha256_file(manifest_path)),
        "release_lineage_id": (lineage["release_lineage_id"]),
        "release_fingerprint": (lineage["release_fingerprint"]),
        "data_run_id": (lineage["data_run_id"]),
        "dataset_fingerprint": (lineage["dataset_fingerprint"]),
    }

    _write_json(
        releases.MODEL_RELEASE_POINTER_PATH,
        pointer,
    )

    resolved = releases.resolve_current_model_release()

    return (
        resolved,
        pointer,
        release_dir,
    )


def test_resolve_current_model_release_happy_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    resolved, pointer, release_dir = _build_valid_release(
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
    )

    assert resolved.model_release_id == pointer["model_release_id"]

    assert resolved.release_dir == release_dir.resolve()

    assert resolved.neumf_model_path == release_dir / "final_neumf.pt"

    assert resolved.two_tower_model_path == release_dir / "final_two_tower.pt"

    assert (
        resolved.item_embeddings_path
        == release_dir / "final_two_tower_item_embeddings.pt"
    )

    assert resolved.lineage["data_run_id"] == "data-run-001"


def test_artifact_corruption_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    resolved, _, _ = _build_valid_release(
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
    )

    resolved.item_embeddings_path.write_bytes(b"bad-embeddings!")

    with pytest.raises(
        RuntimeError,
        match="fingerprint mismatch",
    ):
        releases.resolve_current_model_release()


def test_manifest_corruption_is_rejected_by_pointer_hash(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    resolved, _, _ = _build_valid_release(
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
    )

    manifest = json.loads(resolved.manifest_path.read_text(encoding="utf-8"))

    manifest["catalog"]["n_users"] = 999

    _write_json(
        resolved.manifest_path,
        manifest,
    )

    with pytest.raises(
        RuntimeError,
        match="manifest fingerprint mismatch",
    ):
        releases.resolve_current_model_release()


def test_pointer_path_traversal_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _, pointer, _ = _build_valid_release(
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
    )

    pointer["model_release_dir"] = "../outside-release"

    _write_json(
        releases.MODEL_RELEASE_POINTER_PATH,
        pointer,
    )

    with pytest.raises(
        ValueError,
        match="escapes the model artifact root",
    ):
        releases.resolve_current_model_release()


def test_pointer_lineage_mismatch_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _, pointer, _ = _build_valid_release(
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
    )

    pointer["data_run_id"] = "different-data-run"

    _write_json(
        releases.MODEL_RELEASE_POINTER_PATH,
        pointer,
    )

    with pytest.raises(
        RuntimeError,
        match="pointer lineage does not match",
    ):
        releases.resolve_current_model_release()


def test_incomplete_staging_release_does_not_affect_current_release(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    resolved, pointer, _ = _build_valid_release(
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
    )

    staging_dir = releases.MODEL_RELEASES_DIR / ".staging-crashed"

    staging_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (staging_dir / "partial-checkpoint.pt").write_bytes(b"partial")

    resolved_after_failure = releases.resolve_current_model_release()

    assert resolved_after_failure.model_release_id == resolved.model_release_id

    persisted_pointer = json.loads(
        releases.MODEL_RELEASE_POINTER_PATH.read_text(encoding="utf-8")
    )

    assert persisted_pointer == pointer
