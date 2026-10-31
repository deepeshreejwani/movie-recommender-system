from __future__ import annotations

import json
import os
import shutil
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch

from recsys.config import (
    ARTIFACT_MANIFEST_PATH,
    COLD_ITEM_INDEX,
    FINAL_TRAINING_PROTOCOL,
    ITEM_MAPPING_PATH,
    MOVIE_METADATA_PATH,
    PREPROCESSING_PROTOCOL_VERSION,
    PROCESSED_ARTIFACT_MANIFEST_VERSION,
    SERVING_DIR,
    SERVING_MOVIE_METADATA_PATH,
    SERVING_SEEN_INDICES_PATH,
    SERVING_SEEN_INDPTR_PATH,
    SERVING_USER_MAPPING_PATH,
    TRAIN_PATH,
    USER_MAPPING_PATH,
    VALIDATION_PATH,
)
from recsys.data.loader import (
    load_json,
    load_parquet,
    save_json,
    save_parquet,
)
from recsys.evaluation.evaluator import build_seen_items_csr
from recsys.utils.artifacts import (
    build_model_release_lineage,
    canonical_id_mapping_sha256,
    canonical_interaction_pairs_sha256,
    canonical_json_sha256,
    file_artifact_metadata,
    sha256_file,
    validate_model_data_identity,
    validate_processed_data_identity,
)
from recsys.utils.artifacts import (
    require_metadata_dict as _require_dict,
)
from recsys.utils.artifacts import (
    require_metadata_int as _require_int,
)
from recsys.utils.artifacts import (
    require_metadata_list as _require_list,
)
from recsys.utils.artifacts import (
    require_metadata_string as _require_string,
)
from recsys.utils.artifacts import (
    validate_processed_manifest_file_hash as _validate_manifest_file_hash,
)
from recsys.utils.model_io import (
    CHECKPOINT_VERSION,
    load_model_checkpoint,
    load_tensor,
)
from recsys.utils.releases import (
    SERVING_RELEASE_MANIFEST_NAME,
    ModelRelease,
    resolve_current_model_release,
)

SERVING_BUNDLE_VERSION = 2
SERVING_RELEASE_POINTER_VERSION = 1

SERVING_RELEASES_DIR = SERVING_DIR / "releases"
SERVING_RELEASE_POINTER_PATH = SERVING_DIR / "current_release.json"

SERVING_USER_MAPPING_NAME = SERVING_USER_MAPPING_PATH.name
SERVING_MOVIE_METADATA_NAME = SERVING_MOVIE_METADATA_PATH.name
SERVING_SEEN_INDPTR_NAME = SERVING_SEEN_INDPTR_PATH.name
SERVING_SEEN_INDICES_NAME = SERVING_SEEN_INDICES_PATH.name


def validate_mapping(
    mapping: pl.DataFrame,
    raw_id_col: str,
    index_col: str,
    mapping_name: str,
) -> None:
    """Validate a one-to-one contiguous zero-based ID mapping."""

    if mapping.is_empty():
        raise ValueError(f"{mapping_name} cannot be empty.")

    required_columns = {
        raw_id_col,
        index_col,
    }

    missing_columns = required_columns - set(mapping.columns)

    if missing_columns:
        raise ValueError(
            f"{mapping_name} is missing columns: {sorted(missing_columns)}"
        )

    for column in required_columns:
        if mapping[column].null_count() > 0:
            raise ValueError(f"{mapping_name} contains null values in '{column}'.")

        if not mapping[column].dtype.is_integer():
            raise TypeError(f"{mapping_name} column '{column}' must be integer typed.")

    if mapping[raw_id_col].n_unique() != mapping.height:
        raise ValueError(f"{mapping_name} contains duplicate {raw_id_col} values.")

    if mapping[index_col].n_unique() != mapping.height:
        raise ValueError(f"{mapping_name} contains duplicate {index_col} values.")

    ordered_indices = mapping.sort(index_col)[index_col].cast(pl.Int64).to_numpy()

    expected_indices = np.arange(
        mapping.height,
        dtype=np.int64,
    )

    if not np.array_equal(
        ordered_indices,
        expected_indices,
    ):
        raise ValueError(f"{mapping_name} must contain contiguous zero-based indices.")


def validate_movie_metadata_identity(
    item_mapping: pl.DataFrame,
    movies: pl.DataFrame,
) -> None:
    """
    Prove that movie metadata preserves the exact catalog identity:

        item_idx -> movieId
    """

    required_movie_columns = {
        "item_idx",
        "movieId",
        "title",
        "genres",
    }

    missing_columns = required_movie_columns - set(movies.columns)

    if missing_columns:
        raise ValueError(
            f"Movie metadata is missing columns: {sorted(missing_columns)}"
        )

    if movies.height != item_mapping.height:
        raise ValueError("Movie metadata row count does not match the item mapping.")

    for column in required_movie_columns:
        if movies[column].null_count() > 0:
            raise ValueError(f"Movie metadata contains null values in '{column}'.")

    if movies["item_idx"].n_unique() != movies.height:
        raise ValueError("Movie metadata contains duplicate item_idx values.")

    if movies["movieId"].n_unique() != movies.height:
        raise ValueError("Movie metadata contains duplicate movieId values.")

    expected_pairs = item_mapping.select(
        "item_idx",
        "movieId",
    ).sort("item_idx")

    actual_pairs = movies.select(
        "item_idx",
        "movieId",
    ).sort("item_idx")

    if not expected_pairs.equals(actual_pairs):
        raise RuntimeError(
            "Movie metadata does not preserve the exact item_idx -> movieId mapping."
        )


def validate_interactions(
    interactions: pl.DataFrame,
    frame_name: str,
    n_users: int,
    n_items: int,
    allow_cold_items: bool,
) -> None:
    """Validate encoded interaction bounds."""

    required_columns = {
        "user_idx",
        "item_idx",
    }

    missing_columns = required_columns - set(interactions.columns)

    if missing_columns:
        raise ValueError(f"{frame_name} is missing columns: {sorted(missing_columns)}")

    if interactions.is_empty():
        raise ValueError(f"{frame_name} cannot be empty.")

    if interactions["user_idx"].null_count() > 0:
        raise ValueError(f"{frame_name} contains null user indices.")

    if interactions["item_idx"].null_count() > 0:
        raise ValueError(f"{frame_name} contains null item indices.")

    if interactions.filter(pl.col("user_idx") < 0).height > 0:
        raise ValueError(f"{frame_name} contains negative user indices.")

    if interactions.filter(pl.col("user_idx") >= n_users).height > 0:
        raise ValueError(
            f"{frame_name} contains user indices outside the user vocabulary."
        )

    minimum_item_index = COLD_ITEM_INDEX if allow_cold_items else 0

    if interactions.filter(pl.col("item_idx") < minimum_item_index).height > 0:
        raise ValueError(
            f"{frame_name} contains item indices below {minimum_item_index}."
        )

    if interactions.filter(pl.col("item_idx") >= n_items).height > 0:
        raise ValueError(
            f"{frame_name} contains item indices outside the item vocabulary."
        )


def load_and_validate_processed_lineage(
    validation: pl.DataFrame,
    user_mapping: pl.DataFrame,
    item_mapping: pl.DataFrame,
) -> tuple[
    dict[str, str],
    dict[str, Any],
]:
    """
    Validate processed artifacts and reconstruct the exact lineage
    expected in every final model and embedding artifact.
    """

    manifest = load_json(ARTIFACT_MANIFEST_PATH)

    manifest_version = manifest.get("manifest_version")

    if manifest_version != PROCESSED_ARTIFACT_MANIFEST_VERSION:
        raise RuntimeError(
            "Unsupported processed artifact manifest version: "
            f"expected={PROCESSED_ARTIFACT_MANIFEST_VERSION}, "
            f"actual={manifest_version}."
        )

    protocol_version = _require_string(
        manifest,
        "preprocessing_protocol_version",
        "artifact_manifest",
    )

    if protocol_version != PREPROCESSING_PROTOCOL_VERSION:
        raise RuntimeError(
            "Processed-data preprocessing protocol does not match "
            "the current project configuration."
        )

    identity = {
        "data_run_id": manifest.get("data_run_id"),
        "dataset_fingerprint": manifest.get("dataset_fingerprint"),
        "identity": manifest.get("identity"),
    }

    validate_processed_data_identity(identity)

    identity_payload = _require_dict(
        identity["identity"],
        "identity",
    )

    identity_artifact_hashes = _require_dict(
        identity_payload.get("artifact_hashes"),
        "identity.artifact_hashes",
    )

    processed_artifacts = _require_dict(
        manifest.get("processed_artifacts"),
        "processed_artifacts",
    )

    train_sha256 = _validate_manifest_file_hash(
        processed_artifacts=processed_artifacts,
        artifact_name="train",
        path=TRAIN_PATH,
    )

    validation_sha256 = _validate_manifest_file_hash(
        processed_artifacts=processed_artifacts,
        artifact_name="validation",
        path=VALIDATION_PATH,
    )

    user_mapping_file_sha256 = _validate_manifest_file_hash(
        processed_artifacts=processed_artifacts,
        artifact_name="user_mapping",
        path=USER_MAPPING_PATH,
    )

    item_mapping_file_sha256 = _validate_manifest_file_hash(
        processed_artifacts=processed_artifacts,
        artifact_name="item_mapping",
        path=ITEM_MAPPING_PATH,
    )

    movie_metadata_file_sha256 = _validate_manifest_file_hash(
        processed_artifacts=processed_artifacts,
        artifact_name="movie_metadata",
        path=MOVIE_METADATA_PATH,
    )

    for artifact_name, actual_hash in (
        (
            "train",
            train_sha256,
        ),
        (
            "validation",
            validation_sha256,
        ),
        (
            "user_mapping",
            user_mapping_file_sha256,
        ),
        (
            "item_mapping",
            item_mapping_file_sha256,
        ),
        (
            "movie_metadata",
            movie_metadata_file_sha256,
        ),
    ):
        identity_hash = _require_string(
            identity_artifact_hashes,
            artifact_name,
            "identity.artifact_hashes",
        )

        if actual_hash != identity_hash:
            raise RuntimeError(
                "Processed artifact hash differs between manifest "
                f"lineage sections for '{artifact_name}'."
            )

    semantic_mapping_hashes = _require_dict(
        manifest.get("semantic_mapping_hashes"),
        "semantic_mapping_hashes",
    )

    expected_user_mapping_sha256 = _require_string(
        semantic_mapping_hashes,
        "user_mapping_sha256",
        "semantic_mapping_hashes",
    )

    expected_item_mapping_sha256 = _require_string(
        semantic_mapping_hashes,
        "item_mapping_sha256",
        "semantic_mapping_hashes",
    )

    identity_semantic_hashes = _require_dict(
        identity_payload.get("semantic_mapping_hashes"),
        "identity.semantic_mapping_hashes",
    )

    if (
        _require_string(
            identity_semantic_hashes,
            "user_mapping_sha256",
            "identity.semantic_mapping_hashes",
        )
        != expected_user_mapping_sha256
    ):
        raise RuntimeError(
            "User mapping fingerprint differs between "
            "processed manifest lineage sections."
        )

    if (
        _require_string(
            identity_semantic_hashes,
            "item_mapping_sha256",
            "identity.semantic_mapping_hashes",
        )
        != expected_item_mapping_sha256
    ):
        raise RuntimeError(
            "Item mapping fingerprint differs between "
            "processed manifest lineage sections."
        )

    actual_user_mapping_sha256 = canonical_id_mapping_sha256(
        mapping=user_mapping,
        raw_id_col="userId",
        index_col="user_idx",
    )

    actual_item_mapping_sha256 = canonical_id_mapping_sha256(
        mapping=item_mapping,
        raw_id_col="movieId",
        index_col="item_idx",
    )

    if actual_user_mapping_sha256 != expected_user_mapping_sha256:
        raise RuntimeError(
            "User mapping semantic fingerprint does not "
            "match the processed-data manifest."
        )

    if actual_item_mapping_sha256 != expected_item_mapping_sha256:
        raise RuntimeError(
            "Item mapping semantic fingerprint does not "
            "match the processed-data manifest."
        )

    preprocessing_config = _require_dict(
        manifest.get("preprocessing_config"),
        "preprocessing_config",
    )

    identity_preprocessing_config = _require_dict(
        identity_payload.get("preprocessing_config"),
        "identity.preprocessing_config",
    )

    if preprocessing_config != identity_preprocessing_config:
        raise RuntimeError(
            "Preprocessing configuration differs between "
            "processed manifest lineage sections."
        )

    if preprocessing_config.get("n_users") != user_mapping.height:
        raise RuntimeError(
            "Processed manifest user count does not match the current user mapping."
        )

    if preprocessing_config.get("n_items") != item_mapping.height:
        raise RuntimeError(
            "Processed manifest item count does not match the current item mapping."
        )

    model_data_identity = validate_model_data_identity(
        manifest.get("model_data_identity")
    )

    model_identity_payload = _require_dict(
        model_data_identity["identity"],
        "model_data_identity.identity",
    )

    warm_validation_sha256 = canonical_interaction_pairs_sha256(
        interactions=validation.filter(pl.col("item_idx") >= 0),
    )

    expected_model_identity_fields = {
        "preprocessing_protocol_version": protocol_version,
        "final_training_protocol": FINAL_TRAINING_PROTOCOL,
        "train_sha256": train_sha256,
        "warm_validation_sha256": warm_validation_sha256,
        "user_mapping_sha256": actual_user_mapping_sha256,
        "item_mapping_sha256": actual_item_mapping_sha256,
    }

    for key, expected_value in expected_model_identity_fields.items():
        if model_identity_payload.get(key) != expected_value:
            raise RuntimeError(
                "Model-data identity does not match the exact training inputs "
                f"for '{key}'."
            )

    lineage = build_model_release_lineage(model_data_identity)

    return (
        lineage,
        manifest,
    )


def validate_final_checkpoint(
    checkpoint: dict[str, Any],
    checkpoint_name: str,
    expected_lineage: dict[str, str],
    n_users: int,
    n_items: int,
    final_training_interactions: int,
    warm_validation_interactions: int,
    cold_validation_interactions: int,
) -> None:
    """Validate final-model protocol, counts, vocabulary, and lineage."""

    checkpoint_version = checkpoint.get("checkpoint_version")

    if checkpoint_version != CHECKPOINT_VERSION:
        raise RuntimeError(
            f"{checkpoint_name} checkpoint version mismatch: "
            f"expected={CHECKPOINT_VERSION}, "
            f"actual={checkpoint_version}."
        )

    model_config = _require_dict(
        checkpoint.get("model_config"),
        f"{checkpoint_name}.model_config",
    )

    training_config = _require_dict(
        checkpoint.get("training_config"),
        f"{checkpoint_name}.training_config",
    )

    if (
        _require_int(
            model_config,
            "n_users",
            f"{checkpoint_name}.model_config",
        )
        != n_users
    ):
        raise RuntimeError(
            f"{checkpoint_name} user vocabulary does not match the processed dataset."
        )

    if (
        _require_int(
            model_config,
            "n_items",
            f"{checkpoint_name}.model_config",
        )
        != n_items
    ):
        raise RuntimeError(
            f"{checkpoint_name} item vocabulary does not match the processed dataset."
        )

    training_data = _require_string(
        training_config,
        "training_data",
        f"{checkpoint_name}.training_config",
    )

    if training_data != FINAL_TRAINING_PROTOCOL:
        raise RuntimeError(
            f"{checkpoint_name} training protocol mismatch: "
            f"expected={FINAL_TRAINING_PROTOCOL!r}, "
            f"actual={training_data!r}."
        )

    if (
        _require_int(
            training_config,
            "final_training_interactions",
            f"{checkpoint_name}.training_config",
        )
        != final_training_interactions
    ):
        raise RuntimeError(
            f"{checkpoint_name} final-training interaction count "
            "does not match current processed data."
        )

    if (
        _require_int(
            training_config,
            "warm_validation_interactions",
            f"{checkpoint_name}.training_config",
        )
        != warm_validation_interactions
    ):
        raise RuntimeError(
            f"{checkpoint_name} warm-validation count does "
            "not match current processed data."
        )

    if (
        _require_int(
            training_config,
            "excluded_cold_validation_interactions",
            f"{checkpoint_name}.training_config",
        )
        != cold_validation_interactions
    ):
        raise RuntimeError(
            f"{checkpoint_name} excluded cold-validation count "
            "does not match current processed data."
        )

    if (
        _require_int(
            training_config,
            "cold_item_index",
            f"{checkpoint_name}.training_config",
        )
        != COLD_ITEM_INDEX
    ):
        raise RuntimeError(
            f"{checkpoint_name} cold-item sentinel does not match "
            "project configuration."
        )

    checkpoint_lineage = _require_dict(
        training_config.get("lineage"),
        f"{checkpoint_name}.training_config.lineage",
    )

    if checkpoint_lineage != expected_lineage:
        mismatched_keys = sorted(
            key
            for key in set(checkpoint_lineage) | set(expected_lineage)
            if checkpoint_lineage.get(key) != expected_lineage.get(key)
        )

        raise RuntimeError(
            f"{checkpoint_name} lineage does not match the "
            "processed dataset. Mismatched fields: "
            f"{mismatched_keys}"
        )


def validate_two_tower_embeddings(
    checkpoint: dict[str, Any],
    item_embeddings: torch.Tensor,
    expected_lineage: dict[str, str],
    n_items: int,
    model_release: ModelRelease,
) -> dict[str, Any]:
    """
    Validate persisted Two-Tower item embeddings and prove that
    they belong to the exact immutable model release being served.
    """

    model_config = _require_dict(
        checkpoint.get("model_config"),
        "Two-Tower.model_config",
    )

    embedding_dim = _require_int(
        model_config,
        "embedding_dim",
        "Two-Tower.model_config",
    )

    if item_embeddings.ndim != 2:
        raise RuntimeError("Two-Tower item embeddings must be a 2D tensor.")

    if item_embeddings.shape[0] != n_items:
        raise RuntimeError(
            "Two-Tower item embedding row count does not match "
            "the current item vocabulary."
        )

    if item_embeddings.shape[1] != embedding_dim:
        raise RuntimeError(
            "Two-Tower item embedding dimension does not match the checkpoint."
        )

    if not torch.is_floating_point(item_embeddings):
        raise TypeError("Two-Tower item embeddings must use a floating dtype.")

    if not torch.isfinite(item_embeddings).all():
        raise RuntimeError("Two-Tower item embeddings contain non-finite values.")

    metadata = load_json(model_release.embedding_metadata_path)

    if metadata.get("metadata_version") != 1:
        raise RuntimeError("Unsupported Two-Tower embedding metadata version.")

    if metadata.get("model") != "TwoTowerRecommender":
        raise RuntimeError(
            "Two-Tower embedding metadata contains an unexpected model identifier."
        )

    parent_checkpoint = _require_dict(
        metadata.get("parent_checkpoint"),
        "embedding_metadata.parent_checkpoint",
    )

    expected_checkpoint_sha256 = sha256_file(model_release.two_tower_model_path)

    if (
        _require_string(
            parent_checkpoint,
            "sha256",
            "embedding_metadata.parent_checkpoint",
        )
        != expected_checkpoint_sha256
    ):
        raise RuntimeError(
            "Two-Tower item embeddings are not bound to "
            "the Two-Tower checkpoint in the current model release."
        )

    item_metadata = _require_dict(
        metadata.get("item_embeddings"),
        "embedding_metadata.item_embeddings",
    )

    expected_embedding_sha256 = sha256_file(model_release.item_embeddings_path)

    if (
        _require_string(
            item_metadata,
            "sha256",
            "embedding_metadata.item_embeddings",
        )
        != expected_embedding_sha256
    ):
        raise RuntimeError(
            "Two-Tower item embedding fingerprint does not match its metadata."
        )

    expected_shape = _require_list(
        item_metadata,
        "shape",
        "embedding_metadata.item_embeddings",
    )

    if expected_shape != list(item_embeddings.shape):
        raise RuntimeError(
            "Two-Tower item embedding shape does not match its metadata."
        )

    metadata_lineage = _require_dict(
        metadata.get("lineage"),
        "embedding_metadata.lineage",
    )

    if metadata_lineage != expected_lineage:
        mismatched_keys = sorted(
            key
            for key in set(metadata_lineage) | set(expected_lineage)
            if metadata_lineage.get(key) != expected_lineage.get(key)
        )

        raise RuntimeError(
            "Two-Tower embedding lineage does not match the "
            "processed dataset. Mismatched fields: "
            f"{mismatched_keys}"
        )

    return metadata


def _fsync_directory(
    directory: Path,
) -> None:
    """Best-effort durability for an atomic directory-entry update."""

    try:
        descriptor = os.open(
            directory,
            os.O_RDONLY,
        )

    except OSError:
        return

    try:
        os.fsync(descriptor)

    except OSError:
        pass

    finally:
        os.close(descriptor)


def _atomic_save_json(
    data: dict[str, Any],
    path: Path,
) -> None:
    """Atomically replace one JSON pointer file."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )

    temporary_path = Path(temporary_name)

    try:
        with os.fdopen(
            file_descriptor,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                data,
                file,
                indent=4,
                sort_keys=True,
                allow_nan=False,
            )

            file.write("\n")
            file.flush()
            os.fsync(file.fileno())

        os.replace(
            temporary_path,
            path,
        )

        _fsync_directory(path.parent)

    except Exception:
        with suppress(OSError):
            temporary_path.unlink(missing_ok=True)

        raise


def _bundle_artifact_metadata(
    path: Path,
) -> dict[str, int | str]:
    """Build release-local metadata for one serving-owned artifact."""

    metadata = file_artifact_metadata(path)

    return {
        "filename": path.name,
        "size_bytes": int(metadata["size_bytes"]),
        "sha256": str(metadata["sha256"]),
    }


def _validate_release_filename(
    filename: str,
    field_name: str,
) -> None:
    """Require a plain filename with no absolute path or traversal."""

    candidate = Path(filename)

    if candidate.is_absolute() or candidate.name != filename:
        raise ValueError(f"{field_name} must be a plain filename.")


def validate_staged_serving_bundle(
    release_dir: Path,
    manifest: dict[str, Any],
) -> None:
    """
    Validate the complete staged serving generation before publication.
    """

    if not release_dir.is_dir():
        raise FileNotFoundError(
            f"Serving release directory does not exist: {release_dir}"
        )

    if manifest.get("serving_bundle_version") != SERVING_BUNDLE_VERSION:
        raise RuntimeError(
            "Staged serving bundle version does not match the supported version."
        )

    runtime_artifacts = _require_dict(
        manifest.get("runtime_artifacts"),
        "serving_manifest.runtime_artifacts",
    )

    expected_names = {
        "user_mapping": SERVING_USER_MAPPING_NAME,
        "movie_metadata": SERVING_MOVIE_METADATA_NAME,
        "seen_indptr": SERVING_SEEN_INDPTR_NAME,
        "seen_indices": SERVING_SEEN_INDICES_NAME,
    }

    for artifact_name, expected_filename in expected_names.items():
        metadata = _require_dict(
            runtime_artifacts.get(artifact_name),
            (f"serving_manifest.runtime_artifacts[{artifact_name!r}]"),
        )

        filename = _require_string(
            metadata,
            "filename",
            (f"serving_manifest.runtime_artifacts[{artifact_name!r}]"),
        )

        _validate_release_filename(
            filename,
            (f"serving_manifest.runtime_artifacts[{artifact_name!r}].filename"),
        )

        if filename != expected_filename:
            raise RuntimeError(
                f"Unexpected serving release filename for '{artifact_name}'."
            )

        artifact_path = release_dir / filename

        if not artifact_path.is_file():
            raise FileNotFoundError(
                f"Staged serving artifact does not exist: {artifact_path}"
            )

        expected_size = _require_int(
            metadata,
            "size_bytes",
            (f"serving_manifest.runtime_artifacts[{artifact_name!r}]"),
        )

        if artifact_path.stat().st_size != expected_size:
            raise RuntimeError(
                f"Staged serving artifact size mismatch for '{artifact_name}'."
            )

        expected_sha256 = _require_string(
            metadata,
            "sha256",
            (f"serving_manifest.runtime_artifacts[{artifact_name!r}]"),
        )

        if sha256_file(artifact_path) != expected_sha256:
            raise RuntimeError(
                f"Staged serving artifact fingerprint mismatch for '{artifact_name}'."
            )

    manifest_path = release_dir / SERVING_RELEASE_MANIFEST_NAME

    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Staged serving manifest does not exist: {manifest_path}"
        )

    persisted_manifest = load_json(manifest_path)

    if persisted_manifest != manifest:
        raise RuntimeError(
            "Persisted serving manifest differs from the validated in-memory manifest."
        )


def build_serving_release_identity(
    *,
    lineage: dict[str, str],
    model_release: ModelRelease,
    runtime_artifacts: dict[str, dict[str, int | str]],
    model_artifacts: dict[str, str],
) -> tuple[
    str,
    str,
]:
    """Create a content-derived immutable serving release identity."""

    identity_payload = {
        "serving_bundle_version": SERVING_BUNDLE_VERSION,
        "lineage": dict(lineage),
        "model_release_id": (model_release.model_release_id),
        "model_release_fingerprint": (model_release.model_release_fingerprint),
        "runtime_artifact_hashes": {
            key: str(value["sha256"])
            for key, value in sorted(runtime_artifacts.items())
        },
        "model_artifact_hashes": dict(sorted(model_artifacts.items())),
    }

    serving_release_fingerprint = canonical_json_sha256(identity_payload)

    serving_release_id = f"serving-{serving_release_fingerprint[:20]}"

    return (
        serving_release_id,
        serving_release_fingerprint,
    )


def publish_serving_release(
    *,
    staging_dir: Path,
    serving_release_id: str,
    serving_release_fingerprint: str,
    lineage: dict[str, str],
    model_release: ModelRelease,
) -> Path:
    """
    Atomically publish a fully validated immutable serving generation.

    The previous current-release pointer remains untouched until the
    new directory has been completely published.
    """

    final_release_dir = SERVING_RELEASES_DIR / serving_release_id

    staged_manifest_path = staging_dir / SERVING_RELEASE_MANIFEST_NAME

    staged_manifest_sha256 = sha256_file(staged_manifest_path)

    if final_release_dir.exists():
        existing_manifest_path = final_release_dir / SERVING_RELEASE_MANIFEST_NAME

        if (
            not existing_manifest_path.is_file()
            or sha256_file(existing_manifest_path) != staged_manifest_sha256
        ):
            raise RuntimeError(
                "Serving release ID already exists with "
                "different contents: "
                f"{final_release_dir}"
            )

        shutil.rmtree(staging_dir)

    else:
        os.replace(
            staging_dir,
            final_release_dir,
        )

        _fsync_directory(SERVING_RELEASES_DIR)

    pointer = {
        "pointer_version": (SERVING_RELEASE_POINTER_VERSION),
        "serving_release_id": (serving_release_id),
        "serving_release_fingerprint": (serving_release_fingerprint),
        "serving_release_dir": str(Path("releases") / serving_release_id),
        "manifest_filename": (SERVING_RELEASE_MANIFEST_NAME),
        "manifest_sha256": (staged_manifest_sha256),
        "model_release_id": (model_release.model_release_id),
        "model_release_fingerprint": (model_release.model_release_fingerprint),
        "release_lineage_id": (lineage["release_lineage_id"]),
        "data_run_id": (lineage["data_run_id"]),
    }

    _atomic_save_json(
        pointer,
        SERVING_RELEASE_POINTER_PATH,
    )

    return final_release_dir


def main() -> None:
    # ------------------------------------------------------------------
    # Load processed artifacts
    # ------------------------------------------------------------------

    train = load_parquet(TRAIN_PATH).select(
        "user_idx",
        "item_idx",
    )

    validation = load_parquet(VALIDATION_PATH).select(
        "user_idx",
        "item_idx",
    )

    user_mapping = load_parquet(USER_MAPPING_PATH)

    item_mapping = load_parquet(ITEM_MAPPING_PATH)

    movies = load_parquet(MOVIE_METADATA_PATH)

    # ------------------------------------------------------------------
    # Validate mapping and metadata identity
    # ------------------------------------------------------------------

    validate_mapping(
        mapping=user_mapping,
        raw_id_col="userId",
        index_col="user_idx",
        mapping_name="user_mapping",
    )

    validate_mapping(
        mapping=item_mapping,
        raw_id_col="movieId",
        index_col="item_idx",
        mapping_name="item_mapping",
    )

    n_users = user_mapping.height
    n_items = item_mapping.height

    validate_movie_metadata_identity(
        item_mapping=item_mapping,
        movies=movies,
    )

    # ------------------------------------------------------------------
    # Validate processed provenance and reconstruct exact model lineage
    # ------------------------------------------------------------------

    (
        expected_lineage,
        processed_manifest,
    ) = load_and_validate_processed_lineage(
        validation=validation,
        user_mapping=user_mapping,
        item_mapping=item_mapping,
    )

    model_release = resolve_current_model_release()

    if model_release.lineage != expected_lineage:
        mismatched_keys = sorted(
            key
            for key in set(model_release.lineage) | set(expected_lineage)
            if model_release.lineage.get(key) != expected_lineage.get(key)
        )

        raise RuntimeError(
            "Current model release lineage does not match the "
            "processed dataset. Mismatched fields: "
            f"{mismatched_keys}"
        )

    model_release_catalog = _require_dict(
        model_release.manifest.get("catalog"),
        "model_release_manifest.catalog",
    )

    if (
        _require_int(
            model_release_catalog,
            "n_users",
            "model_release_manifest.catalog",
        )
        != n_users
    ):
        raise RuntimeError(
            "Current model release user vocabulary does not "
            "match the processed dataset."
        )

    if (
        _require_int(
            model_release_catalog,
            "n_items",
            "model_release_manifest.catalog",
        )
        != n_items
    ):
        raise RuntimeError(
            "Current model release item vocabulary does not "
            "match the processed dataset."
        )

    # ------------------------------------------------------------------
    # Validate train / validation data
    # ------------------------------------------------------------------

    validate_interactions(
        interactions=train,
        frame_name="train",
        n_users=n_users,
        n_items=n_items,
        allow_cold_items=False,
    )

    validate_interactions(
        interactions=validation,
        frame_name="validation",
        n_users=n_users,
        n_items=n_items,
        allow_cold_items=True,
    )

    # ------------------------------------------------------------------
    # Build final-model seen history
    #
    # Final models were trained on:
    #
    #     train + warm validation
    #
    # Cold validation targets are outside the frozen vocabulary.
    # ------------------------------------------------------------------

    cold_validation_count = validation.filter(
        pl.col("item_idx") == COLD_ITEM_INDEX
    ).height

    warm_validation = validation.filter(pl.col("item_idx") >= 0)

    warm_validation_count = warm_validation.height

    final_training_interactions = train.height + warm_validation_count

    observed = pl.concat(
        [
            train,
            warm_validation,
        ],
        how="vertical",
    ).unique(
        subset=[
            "user_idx",
            "item_idx",
        ]
    )

    validate_interactions(
        interactions=observed,
        frame_name="serving_seen_history",
        n_users=n_users,
        n_items=n_items,
        allow_cold_items=False,
    )

    seen_items = build_seen_items_csr(
        interactions=observed,
        n_users=n_users,
    )

    if seen_items.indptr.size != n_users + 1:
        raise RuntimeError("Seen-item CSR indptr has an invalid size.")

    if seen_items.indptr[-1] != seen_items.indices.size:
        raise RuntimeError("Seen-item CSR structure is inconsistent.")

    if seen_items.indices.size > 0 and (
        seen_items.indices.min() < 0 or seen_items.indices.max() >= n_items
    ):
        raise RuntimeError("Seen-item CSR contains invalid item indices.")

    # ------------------------------------------------------------------
    # Load and validate FINAL model artifacts before writing anything
    # into the serving directory.
    # ------------------------------------------------------------------

    two_tower_checkpoint = load_model_checkpoint(
        model_release.two_tower_model_path,
        device="cpu",
    )

    neumf_checkpoint = load_model_checkpoint(
        model_release.neumf_model_path,
        device="cpu",
    )

    item_embeddings = load_tensor(
        model_release.item_embeddings_path,
        device="cpu",
    )

    validate_final_checkpoint(
        checkpoint=two_tower_checkpoint,
        checkpoint_name="Two-Tower",
        expected_lineage=expected_lineage,
        n_users=n_users,
        n_items=n_items,
        final_training_interactions=(final_training_interactions),
        warm_validation_interactions=(warm_validation_count),
        cold_validation_interactions=(cold_validation_count),
    )

    validate_final_checkpoint(
        checkpoint=neumf_checkpoint,
        checkpoint_name="NeuMF",
        expected_lineage=expected_lineage,
        n_users=n_users,
        n_items=n_items,
        final_training_interactions=(final_training_interactions),
        warm_validation_interactions=(warm_validation_count),
        cold_validation_interactions=(cold_validation_count),
    )

    embedding_metadata = validate_two_tower_embeddings(
        checkpoint=two_tower_checkpoint,
        item_embeddings=item_embeddings,
        expected_lineage=expected_lineage,
        n_items=n_items,
        model_release=model_release,
    )

    two_tower_checkpoint_sha256 = sha256_file(model_release.two_tower_model_path)

    neumf_checkpoint_sha256 = sha256_file(model_release.neumf_model_path)

    item_embeddings_sha256 = sha256_file(model_release.item_embeddings_path)

    # ------------------------------------------------------------------
    # Build an immutable serving release in a staging directory.
    # Nothing becomes live until the complete bundle validates.
    # ------------------------------------------------------------------

    SERVING_RELEASES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=".staging-",
            dir=SERVING_RELEASES_DIR,
        )
    )

    try:
        staged_user_mapping_path = staging_dir / SERVING_USER_MAPPING_NAME

        staged_movie_metadata_path = staging_dir / SERVING_MOVIE_METADATA_NAME

        staged_seen_indptr_path = staging_dir / SERVING_SEEN_INDPTR_NAME

        staged_seen_indices_path = staging_dir / SERVING_SEEN_INDICES_NAME

        staged_manifest_path = staging_dir / SERVING_RELEASE_MANIFEST_NAME

        save_parquet(
            user_mapping,
            staged_user_mapping_path,
        )

        save_parquet(
            movies,
            staged_movie_metadata_path,
        )

        np.save(
            staged_seen_indptr_path,
            seen_items.indptr.astype(
                np.int64,
                copy=False,
            ),
        )

        np.save(
            staged_seen_indices_path,
            seen_items.indices.astype(
                np.int32,
                copy=False,
            ),
        )

        runtime_artifacts = {
            "user_mapping": (_bundle_artifact_metadata(staged_user_mapping_path)),
            "movie_metadata": (_bundle_artifact_metadata(staged_movie_metadata_path)),
            "seen_indptr": (_bundle_artifact_metadata(staged_seen_indptr_path)),
            "seen_indices": (_bundle_artifact_metadata(staged_seen_indices_path)),
        }

        model_artifacts = {
            "two_tower_checkpoint_sha256": (two_tower_checkpoint_sha256),
            "neumf_checkpoint_sha256": (neumf_checkpoint_sha256),
            "two_tower_item_embeddings_sha256": (item_embeddings_sha256),
            "two_tower_embedding_metadata_sha256": (
                sha256_file(model_release.embedding_metadata_path)
            ),
            "model_release_manifest_sha256": (sha256_file(model_release.manifest_path)),
        }

        (
            serving_release_id,
            serving_release_fingerprint,
        ) = build_serving_release_identity(
            lineage=expected_lineage,
            model_release=model_release,
            runtime_artifacts=runtime_artifacts,
            model_artifacts=model_artifacts,
        )

        processed_semantic_hashes = _require_dict(
            processed_manifest.get("semantic_mapping_hashes"),
            "processed_manifest.semantic_mapping_hashes",
        )

        manifest = {
            "serving_bundle_version": (SERVING_BUNDLE_VERSION),
            "serving_release": {
                "serving_release_id": (serving_release_id),
                "serving_release_fingerprint": (serving_release_fingerprint),
            },
            "release": {
                "release_lineage_id": (expected_lineage["release_lineage_id"]),
                "release_fingerprint": (expected_lineage["release_fingerprint"]),
                "data_run_id": (expected_lineage["data_run_id"]),
                "dataset_fingerprint": (expected_lineage["dataset_fingerprint"]),
                "model_release_id": (model_release.model_release_id),
                "model_release_fingerprint": (model_release.model_release_fingerprint),
                "model_release_manifest_sha256": (
                    model_artifacts["model_release_manifest_sha256"]
                ),
            },
            "lineage": dict(expected_lineage),
            "catalog": {
                "n_users": n_users,
                "n_items": n_items,
                "cold_item_index": (COLD_ITEM_INDEX),
                "user_mapping_sha256": (expected_lineage["user_mapping_sha256"]),
                "item_mapping_sha256": (expected_lineage["item_mapping_sha256"]),
            },
            "training_history": {
                "source": (FINAL_TRAINING_PROTOCOL),
                "train_interactions": (train.height),
                "warm_validation_interactions": (warm_validation_count),
                "excluded_cold_validation_interactions": (cold_validation_count),
                "final_training_interactions": (final_training_interactions),
                "unique_seen_user_item_pairs": (seen_items.indices.size),
            },
            "model_artifacts": (model_artifacts),
            "embedding_binding": {
                "parent_checkpoint_sha256": (
                    _require_string(
                        _require_dict(
                            embedding_metadata.get("parent_checkpoint"),
                            ("embedding_metadata.parent_checkpoint"),
                        ),
                        "sha256",
                        ("embedding_metadata.parent_checkpoint"),
                    )
                ),
                "item_embeddings_sha256": (
                    _require_string(
                        _require_dict(
                            embedding_metadata.get("item_embeddings"),
                            ("embedding_metadata.item_embeddings"),
                        ),
                        "sha256",
                        ("embedding_metadata.item_embeddings"),
                    )
                ),
            },
            "runtime_artifacts": (runtime_artifacts),
            "source_vocabulary": {
                "user_mapping": (file_artifact_metadata(USER_MAPPING_PATH)),
                "item_mapping": (file_artifact_metadata(ITEM_MAPPING_PATH)),
                "movie_metadata": (file_artifact_metadata(MOVIE_METADATA_PATH)),
                "semantic_mapping_hashes": {
                    "user_mapping_sha256": (
                        _require_string(
                            processed_semantic_hashes,
                            "user_mapping_sha256",
                            ("processed_manifest.semantic_mapping_hashes"),
                        )
                    ),
                    "item_mapping_sha256": (
                        _require_string(
                            processed_semantic_hashes,
                            "item_mapping_sha256",
                            ("processed_manifest.semantic_mapping_hashes"),
                        )
                    ),
                },
                "item_mapping_movie_metadata_identity": (
                    "validated_exact_item_idx_to_movieId"
                ),
            },
        }

        save_json(
            manifest,
            staged_manifest_path,
        )

        validate_staged_serving_bundle(
            release_dir=staging_dir,
            manifest=manifest,
        )

        final_serving_release_dir = publish_serving_release(
            staging_dir=staging_dir,
            serving_release_id=serving_release_id,
            serving_release_fingerprint=(serving_release_fingerprint),
            lineage=expected_lineage,
            model_release=model_release,
        )

    except Exception:
        if staging_dir.exists():
            shutil.rmtree(
                staging_dir,
                ignore_errors=True,
            )

        raise

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    print(f"Release lineage          : {expected_lineage['release_lineage_id']}")

    print(f"Model release            : {model_release.model_release_id}")

    print(f"Serving release          : {serving_release_id}")

    print(f"Data run ID              : {expected_lineage['data_run_id']}")

    print(f"Users                    : {n_users:,}")

    print(f"Items                    : {n_items:,}")

    print(f"Train interactions       : {train.height:,}")

    print(f"Warm validation added    : {warm_validation_count:,}")

    print(f"Cold validation excluded : {cold_validation_count:,}")

    print(f"Final training rows       : {final_training_interactions:,}")

    print(f"Unique seen pairs        : {seen_items.indices.size:,}")

    print(f"\nServing release directory: {final_serving_release_dir}")

    print(f"Serving pointer          : {SERVING_RELEASE_POINTER_PATH}")


if __name__ == "__main__":
    main()
