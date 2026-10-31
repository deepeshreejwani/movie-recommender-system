from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from recsys.config import (
    FINAL_NEUMF_MODEL_PATH,
)
from recsys.data.loader import load_json
from recsys.utils.artifacts import (
    canonical_json_sha256,
    sha256_file,
)
from recsys.utils.artifacts import (
    require_metadata_dict as _require_dict,
)
from recsys.utils.artifacts import (
    require_metadata_int as _require_int,
)
from recsys.utils.artifacts import (
    require_metadata_string as _require_string,
)

MODEL_RELEASE_MANIFEST_VERSION = 1
MODEL_RELEASE_POINTER_VERSION = 1

MODEL_RELEASES_DIR = FINAL_NEUMF_MODEL_PATH.parent / "releases"

MODEL_RELEASE_POINTER_PATH = FINAL_NEUMF_MODEL_PATH.parent / "current_release.json"

MODEL_RELEASE_MANIFEST_NAME = "model_release_manifest.json"

FINAL_TWO_TOWER_EMBEDDING_METADATA_NAME = "final_two_tower_embeddings_metadata.json"

# Shared serving-release contract. Builder and runtime must use this exact filename.
SERVING_RELEASE_MANIFEST_NAME = "serving_manifest.json"


@dataclass(frozen=True)
class ModelRelease:
    """Validated immutable production-model generation."""

    model_release_id: str
    model_release_fingerprint: str
    release_dir: Path
    manifest_path: Path
    manifest: dict[str, Any]
    lineage: dict[str, str]
    neumf_model_path: Path
    two_tower_model_path: Path
    user_embeddings_path: Path
    item_embeddings_path: Path
    embedding_metadata_path: Path


def _resolve_relative_path(
    root: Path,
    relative_path: str,
    field_name: str,
) -> Path:
    """
    Resolve a manifest-controlled relative path while preventing
    absolute paths and traversal outside the expected root.
    """

    relative = Path(relative_path)

    if relative.is_absolute():
        raise ValueError(f"{field_name} must be relative.")

    root_resolved = root.resolve()

    resolved = (root / relative).resolve()

    try:
        resolved.relative_to(root_resolved)

    except ValueError as error:
        raise ValueError(f"{field_name} escapes the model artifact root.") from error

    return resolved


def _resolve_release_artifact(
    release_dir: Path,
    artifact_name: str,
    metadata: dict[str, Any],
) -> Path:
    """Resolve and validate one immutable release artifact."""

    metadata_name = f"model_release.artifacts[{artifact_name!r}]"

    filename = _require_string(
        metadata,
        "filename",
        metadata_name,
    )

    filename_path = Path(filename)

    if filename_path.is_absolute() or filename_path.name != filename:
        raise ValueError(f"{metadata_name} filename must be a plain filename.")

    artifact_path = release_dir / filename

    if not artifact_path.is_file():
        raise FileNotFoundError(
            f"Model release artifact does not exist: {artifact_path}"
        )

    expected_size = _require_int(
        metadata,
        "size_bytes",
        metadata_name,
    )

    actual_size = artifact_path.stat().st_size

    if actual_size != expected_size:
        raise RuntimeError(
            "Model release artifact size mismatch for "
            f"'{artifact_name}'. expected={expected_size}, "
            f"actual={actual_size}."
        )

    expected_sha256 = _require_string(
        metadata,
        "sha256",
        metadata_name,
    )

    actual_sha256 = sha256_file(artifact_path)

    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"Model release artifact fingerprint mismatch for '{artifact_name}'."
        )

    return artifact_path


def resolve_current_model_release() -> ModelRelease:
    """
    Resolve and validate the atomically published current model release.

    The pointer is read once. Every model artifact is then loaded from
    the immutable release directory selected by that pointer.
    """

    pointer = load_json(MODEL_RELEASE_POINTER_PATH)

    pointer_version = pointer.get("pointer_version")

    if pointer_version != MODEL_RELEASE_POINTER_VERSION:
        raise RuntimeError(
            "Unsupported model release pointer version: "
            f"expected={MODEL_RELEASE_POINTER_VERSION}, "
            f"actual={pointer_version}."
        )

    model_release_id = _require_string(
        pointer,
        "model_release_id",
        "model_release_pointer",
    )

    release_dir_value = _require_string(
        pointer,
        "model_release_dir",
        "model_release_pointer",
    )

    release_dir = _resolve_relative_path(
        root=FINAL_NEUMF_MODEL_PATH.parent,
        relative_path=release_dir_value,
        field_name="model_release_pointer.model_release_dir",
    )

    if not release_dir.is_dir():
        raise FileNotFoundError(
            f"Current model release directory does not exist: {release_dir}"
        )

    if release_dir.parent != MODEL_RELEASES_DIR.resolve():
        raise RuntimeError(
            "Current model release does not resolve inside "
            "the configured releases directory."
        )

    if release_dir.name != model_release_id:
        raise RuntimeError(
            "Model release pointer ID does not match the selected release directory."
        )

    manifest_filename = _require_string(
        pointer,
        "manifest_filename",
        "model_release_pointer",
    )

    if (
        Path(manifest_filename).is_absolute()
        or Path(manifest_filename).name != manifest_filename
    ):
        raise ValueError("Model release manifest filename must be a plain filename.")

    if manifest_filename != MODEL_RELEASE_MANIFEST_NAME:
        raise RuntimeError(
            "Model release pointer references an unexpected manifest filename."
        )

    manifest_path = release_dir / manifest_filename

    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Current model release manifest does not exist: {manifest_path}"
        )

    expected_manifest_sha256 = _require_string(
        pointer,
        "manifest_sha256",
        "model_release_pointer",
    )

    actual_manifest_sha256 = sha256_file(manifest_path)

    if actual_manifest_sha256 != expected_manifest_sha256:
        raise RuntimeError("Current model release manifest fingerprint mismatch.")

    manifest = load_json(manifest_path)

    manifest_version = manifest.get("model_release_manifest_version")

    if manifest_version != MODEL_RELEASE_MANIFEST_VERSION:
        raise RuntimeError(
            "Unsupported model release manifest version: "
            f"expected={MODEL_RELEASE_MANIFEST_VERSION}, "
            f"actual={manifest_version}."
        )

    manifest_release_id = _require_string(
        manifest,
        "model_release_id",
        "model_release_manifest",
    )

    if manifest_release_id != model_release_id:
        raise RuntimeError("Model release pointer and manifest IDs do not match.")

    lineage = _require_dict(
        manifest.get("lineage"),
        "model_release_manifest.lineage",
    )

    required_lineage_fields = (
        "release_lineage_id",
        "release_fingerprint",
        "data_run_id",
        "dataset_fingerprint",
    )

    for field_name in required_lineage_fields:
        manifest_value = _require_string(
            lineage,
            field_name,
            "model_release_manifest.lineage",
        )

        pointer_value = _require_string(
            pointer,
            field_name,
            "model_release_pointer",
        )

        if manifest_value != pointer_value:
            raise RuntimeError(
                "Model release pointer lineage does not match "
                f"the release manifest for '{field_name}'."
            )

    artifacts = _require_dict(
        manifest.get("artifacts"),
        "model_release_manifest.artifacts",
    )

    required_artifact_names = {
        "neumf_model",
        "two_tower_model",
        "two_tower_user_embeddings",
        "two_tower_item_embeddings",
        "two_tower_embedding_metadata",
    }

    missing_artifacts = required_artifact_names - set(artifacts)

    if missing_artifacts:
        raise RuntimeError(
            f"Model release manifest is missing artifacts: {sorted(missing_artifacts)}"
        )

    artifact_paths: dict[str, Path] = {}

    artifact_hashes: dict[str, str] = {}

    for artifact_name in sorted(required_artifact_names):
        metadata = _require_dict(
            artifacts.get(artifact_name),
            (f"model_release_manifest.artifacts[{artifact_name!r}]"),
        )

        artifact_path = _resolve_release_artifact(
            release_dir=release_dir,
            artifact_name=artifact_name,
            metadata=metadata,
        )

        artifact_paths[artifact_name] = artifact_path

        artifact_hashes[artifact_name] = _require_string(
            metadata,
            "sha256",
            (f"model_release_manifest.artifacts[{artifact_name!r}]"),
        )

    release_identity = {
        "model_release_manifest_version": (MODEL_RELEASE_MANIFEST_VERSION),
        "lineage": dict(lineage),
        "artifact_hashes": (artifact_hashes),
    }

    expected_release_fingerprint = canonical_json_sha256(release_identity)

    manifest_release_fingerprint = _require_string(
        manifest,
        "model_release_fingerprint",
        "model_release_manifest",
    )

    if manifest_release_fingerprint != expected_release_fingerprint:
        raise RuntimeError(
            "Model release fingerprint does not match its manifest contents."
        )

    expected_release_id = f"model-{expected_release_fingerprint[:20]}"

    if model_release_id != expected_release_id:
        raise RuntimeError(
            "Model release ID does not match the content-derived release fingerprint."
        )

    catalog = _require_dict(
        manifest.get("catalog"),
        "model_release_manifest.catalog",
    )

    n_users = _require_int(
        catalog,
        "n_users",
        "model_release_manifest.catalog",
    )

    n_items = _require_int(
        catalog,
        "n_items",
        "model_release_manifest.catalog",
    )

    if n_users <= 0:
        raise RuntimeError("Model release user vocabulary must be positive.")

    if n_items <= 0:
        raise RuntimeError("Model release item vocabulary must be positive.")

    return ModelRelease(
        model_release_id=model_release_id,
        model_release_fingerprint=(manifest_release_fingerprint),
        release_dir=release_dir,
        manifest_path=manifest_path,
        manifest=manifest,
        lineage={str(key): str(value) for key, value in lineage.items()},
        neumf_model_path=artifact_paths["neumf_model"],
        two_tower_model_path=artifact_paths["two_tower_model"],
        user_embeddings_path=artifact_paths["two_tower_user_embeddings"],
        item_embeddings_path=artifact_paths["two_tower_item_embeddings"],
        embedding_metadata_path=artifact_paths["two_tower_embedding_metadata"],
    )
