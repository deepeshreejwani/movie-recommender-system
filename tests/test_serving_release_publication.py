from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from recsys.utils.artifacts import sha256_file
from scripts import build_serving_artifacts as builder


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


def _configure_serving_paths(
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
) -> None:
    monkeypatch.setattr(
        builder,
        "SERVING_DIR",
        root,
    )

    monkeypatch.setattr(
        builder,
        "SERVING_RELEASES_DIR",
        root / "releases",
    )

    monkeypatch.setattr(
        builder,
        "SERVING_RELEASE_POINTER_PATH",
        root / "current_release.json",
    )


def _fake_model_release() -> SimpleNamespace:
    return SimpleNamespace(
        model_release_id="model-1234567890abcdef1234",
        model_release_fingerprint="a" * 64,
    )


def _lineage() -> dict[str, str]:
    return {
        "release_lineage_id": "release-1234567890abcdef1234",
        "data_run_id": "data-run-001",
    }


def _build_staged_bundle(
    staging_dir: Path,
) -> dict[str, Any]:
    staging_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    files = {
        "user_mapping": (
            builder.SERVING_USER_MAPPING_NAME,
            b"user-mapping",
        ),
        "movie_metadata": (
            builder.SERVING_MOVIE_METADATA_NAME,
            b"movie-metadata",
        ),
        "seen_indptr": (
            builder.SERVING_SEEN_INDPTR_NAME,
            b"seen-indptr",
        ),
        "seen_indices": (
            builder.SERVING_SEEN_INDICES_NAME,
            b"seen-indices",
        ),
    }

    runtime_artifacts: dict[
        str,
        dict[str, int | str],
    ] = {}

    for artifact_name, (
        filename,
        contents,
    ) in files.items():
        path = staging_dir / filename

        path.write_bytes(contents)

        runtime_artifacts[artifact_name] = {
            "filename": filename,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }

    manifest = {
        "serving_bundle_version": (builder.SERVING_BUNDLE_VERSION),
        "runtime_artifacts": (runtime_artifacts),
    }

    _write_json(
        staging_dir / builder.SERVING_RELEASE_MANIFEST_NAME,
        manifest,
    )

    return manifest


def test_staged_serving_bundle_validates(
    tmp_path: Path,
) -> None:
    staging_dir = tmp_path / ".staging-valid"

    manifest = _build_staged_bundle(staging_dir)

    builder.validate_staged_serving_bundle(
        release_dir=staging_dir,
        manifest=manifest,
    )


def test_missing_staged_artifact_is_rejected(
    tmp_path: Path,
) -> None:
    staging_dir = tmp_path / ".staging-missing"

    manifest = _build_staged_bundle(staging_dir)

    (staging_dir / builder.SERVING_SEEN_INDICES_NAME).unlink()

    with pytest.raises(
        FileNotFoundError,
        match="does not exist",
    ):
        builder.validate_staged_serving_bundle(
            release_dir=staging_dir,
            manifest=manifest,
        )


def test_same_size_staged_artifact_corruption_is_rejected(
    tmp_path: Path,
) -> None:
    staging_dir = tmp_path / ".staging-corrupt"

    manifest = _build_staged_bundle(staging_dir)

    path = staging_dir / builder.SERVING_SEEN_INDICES_NAME

    original = path.read_bytes()

    replacement = b"x" * len(original)

    assert replacement != original

    path.write_bytes(replacement)

    with pytest.raises(
        RuntimeError,
        match="fingerprint mismatch",
    ):
        builder.validate_staged_serving_bundle(
            release_dir=staging_dir,
            manifest=manifest,
        )


def test_successful_publication_switches_pointer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    serving_root = tmp_path / "serving"

    _configure_serving_paths(
        monkeypatch=monkeypatch,
        root=serving_root,
    )

    builder.SERVING_RELEASES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    staging_dir = builder.SERVING_RELEASES_DIR / ".staging-success"

    manifest = _build_staged_bundle(staging_dir)

    builder.validate_staged_serving_bundle(
        release_dir=staging_dir,
        manifest=manifest,
    )

    model_release = _fake_model_release()

    lineage = _lineage()

    serving_release_id = "serving-1234567890abcdef1234"

    serving_release_fingerprint = "b" * 64

    final_release_dir = builder.publish_serving_release(
        staging_dir=staging_dir,
        serving_release_id=(serving_release_id),
        serving_release_fingerprint=(serving_release_fingerprint),
        lineage=lineage,
        model_release=model_release,
    )

    assert final_release_dir.is_dir()

    assert not staging_dir.exists()

    pointer = json.loads(
        builder.SERVING_RELEASE_POINTER_PATH.read_text(encoding="utf-8")
    )

    assert pointer["serving_release_id"] == serving_release_id

    assert pointer["serving_release_fingerprint"] == serving_release_fingerprint

    assert pointer["model_release_id"] == model_release.model_release_id

    assert pointer["manifest_sha256"] == sha256_file(
        final_release_dir / builder.SERVING_RELEASE_MANIFEST_NAME
    )


def test_failure_before_pointer_switch_preserves_previous_pointer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    serving_root = tmp_path / "serving"

    _configure_serving_paths(
        monkeypatch=monkeypatch,
        root=serving_root,
    )

    builder.SERVING_RELEASES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    previous_pointer = {
        "pointer_version": (builder.SERVING_RELEASE_POINTER_VERSION),
        "serving_release_id": ("serving-previous"),
    }

    _write_json(
        builder.SERVING_RELEASE_POINTER_PATH,
        previous_pointer,
    )

    staging_dir = builder.SERVING_RELEASES_DIR / ".staging-pointer-failure"

    manifest = _build_staged_bundle(staging_dir)

    builder.validate_staged_serving_bundle(
        release_dir=staging_dir,
        manifest=manifest,
    )

    def fail_pointer_write(
        data: dict[str, Any],
        path: Path,
    ) -> None:
        del data
        del path

        raise OSError("simulated pointer failure")

    monkeypatch.setattr(
        builder,
        "_atomic_save_json",
        fail_pointer_write,
    )

    serving_release_id = "serving-new-release"

    with pytest.raises(
        OSError,
        match="simulated pointer failure",
    ):
        builder.publish_serving_release(
            staging_dir=staging_dir,
            serving_release_id=(serving_release_id),
            serving_release_fingerprint=("c" * 64),
            lineage=_lineage(),
            model_release=_fake_model_release(),
        )

    persisted_pointer = json.loads(
        builder.SERVING_RELEASE_POINTER_PATH.read_text(encoding="utf-8")
    )

    assert persisted_pointer == previous_pointer

    assert (builder.SERVING_RELEASES_DIR / serving_release_id).is_dir()


def test_incomplete_staging_directory_does_not_change_current_pointer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    serving_root = tmp_path / "serving"

    _configure_serving_paths(
        monkeypatch=monkeypatch,
        root=serving_root,
    )

    builder.SERVING_RELEASES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    previous_pointer = {
        "pointer_version": (builder.SERVING_RELEASE_POINTER_VERSION),
        "serving_release_id": ("serving-stable"),
    }

    _write_json(
        builder.SERVING_RELEASE_POINTER_PATH,
        previous_pointer,
    )

    incomplete_staging_dir = builder.SERVING_RELEASES_DIR / ".staging-crashed"

    incomplete_staging_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (incomplete_staging_dir / builder.SERVING_USER_MAPPING_NAME).write_bytes(b"partial")

    persisted_pointer = json.loads(
        builder.SERVING_RELEASE_POINTER_PATH.read_text(encoding="utf-8")
    )

    assert persisted_pointer == previous_pointer


def test_existing_release_with_different_manifest_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    serving_root = tmp_path / "serving"

    _configure_serving_paths(
        monkeypatch=monkeypatch,
        root=serving_root,
    )

    builder.SERVING_RELEASES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    serving_release_id = "serving-collision"

    existing_release_dir = builder.SERVING_RELEASES_DIR / serving_release_id

    existing_release_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    _write_json(
        existing_release_dir / builder.SERVING_RELEASE_MANIFEST_NAME,
        {
            "different": True,
        },
    )

    staging_dir = builder.SERVING_RELEASES_DIR / ".staging-collision"

    manifest = _build_staged_bundle(staging_dir)

    builder.validate_staged_serving_bundle(
        release_dir=staging_dir,
        manifest=manifest,
    )

    with pytest.raises(
        RuntimeError,
        match="already exists with different contents",
    ):
        builder.publish_serving_release(
            staging_dir=staging_dir,
            serving_release_id=(serving_release_id),
            serving_release_fingerprint=("d" * 64),
            lineage=_lineage(),
            model_release=_fake_model_release(),
        )
