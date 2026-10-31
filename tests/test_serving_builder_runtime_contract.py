from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import recsys.serving.inference as serving
from recsys.utils.artifacts import sha256_file
from scripts import build_serving_artifacts as builder


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=4, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_builder_pointer_is_accepted_by_real_serving_resolver(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    serving_root = tmp_path / "serving"
    releases_dir = serving_root / "releases"
    pointer_path = serving_root / "current_release.json"
    releases_dir.mkdir(parents=True)

    monkeypatch.setattr(builder, "SERVING_DIR", serving_root)
    monkeypatch.setattr(builder, "SERVING_RELEASES_DIR", releases_dir)
    monkeypatch.setattr(builder, "SERVING_RELEASE_POINTER_PATH", pointer_path)

    monkeypatch.setattr(serving, "SERVING_DIR", serving_root)
    monkeypatch.setattr(serving, "SERVING_RELEASES_DIR", releases_dir)
    monkeypatch.setattr(serving, "SERVING_RELEASE_POINTER_PATH", pointer_path)

    staging_dir = releases_dir / ".staging-contract"
    staging_dir.mkdir()

    file_specs = {
        "user_mapping": builder.SERVING_USER_MAPPING_NAME,
        "movie_metadata": builder.SERVING_MOVIE_METADATA_NAME,
        "seen_indptr": builder.SERVING_SEEN_INDPTR_NAME,
        "seen_indices": builder.SERVING_SEEN_INDICES_NAME,
    }

    runtime_artifacts: dict[str, dict[str, int | str]] = {}

    for index, (artifact_name, filename) in enumerate(file_specs.items(), start=1):
        path = staging_dir / filename
        path.write_bytes(f"artifact-{index}".encode())
        runtime_artifacts[artifact_name] = {
            "filename": filename,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }

    lineage = {
        "data_run_id": "model-data-123",
        "dataset_fingerprint": "1" * 64,
        "preprocessing_protocol_version": "preprocess-v1",
        "final_training_protocol": "train+warm_validation",
        "train_sha256": "2" * 64,
        "warm_validation_sha256": "3" * 64,
        "user_mapping_sha256": "4" * 64,
        "item_mapping_sha256": "5" * 64,
        "release_lineage_id": "release-123",
        "release_fingerprint": "6" * 64,
    }

    model_release = SimpleNamespace(
        model_release_id="model-123",
        model_release_fingerprint="7" * 64,
    )

    model_artifacts = {
        "model_release_manifest_sha256": "8" * 64,
        "neumf_model_sha256": "9" * 64,
    }

    serving_release_id, serving_release_fingerprint = (
        builder.build_serving_release_identity(
            lineage=lineage,
            model_release=model_release,
            runtime_artifacts=runtime_artifacts,
            model_artifacts=model_artifacts,
        )
    )

    manifest = {
        "serving_bundle_version": builder.SERVING_BUNDLE_VERSION,
        "serving_release": {
            "serving_release_id": serving_release_id,
            "serving_release_fingerprint": serving_release_fingerprint,
        },
        "release": {
            "model_release_id": model_release.model_release_id,
            "model_release_fingerprint": model_release.model_release_fingerprint,
        },
        "lineage": lineage,
        "runtime_artifacts": runtime_artifacts,
        "model_artifacts": model_artifacts,
    }

    _write_json(
        staging_dir / builder.SERVING_RELEASE_MANIFEST_NAME,
        manifest,
    )

    builder.publish_serving_release(
        staging_dir=staging_dir,
        serving_release_id=serving_release_id,
        serving_release_fingerprint=serving_release_fingerprint,
        lineage=lineage,
        model_release=model_release,
    )

    resolved = serving.resolve_current_serving_release()

    assert resolved.serving_release_id == serving_release_id
    assert resolved.manifest_path.name == builder.SERVING_RELEASE_MANIFEST_NAME
