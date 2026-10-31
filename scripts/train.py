from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import tempfile
import time
from contextlib import suppress
from pathlib import Path

import numpy as np
import polars as pl
import torch

from recsys.config import (
    ARTIFACT_MANIFEST_PATH,
    COLD_ITEM_INDEX,
    FINAL_NEUMF_MODEL_PATH,
    FINAL_TRAINING_PROTOCOL,
    FINAL_TWO_TOWER_ITEM_EMBEDDINGS_PATH,
    FINAL_TWO_TOWER_MODEL_PATH,
    FINAL_TWO_TOWER_USER_EMBEDDINGS_PATH,
    ITEM_MAPPING_PATH,
    NCF_EMBEDDING_DIM,
    NCF_EPOCHS,
    NCF_ITEM_SCORE_BATCH_SIZE,
    NCF_LEARNING_RATE,
    NCF_MLP_LAYERS,
    NCF_WEIGHT_DECAY,
    NEUMF_EPOCHS,
    NEUMF_LEARNING_RATE,
    NEUMF_PRETRAIN_ALPHA,
    NEUMF_WEIGHT_DECAY,
    PREPROCESSING_PROTOCOL_VERSION,
    PROCESSED_ARTIFACT_MANIFEST_VERSION,
    RANDOM_SEED,
    RETRIEVAL_EMBEDDING_DIM,
    RETRIEVAL_EPOCHS,
    RETRIEVAL_HIDDEN_DIM,
    RETRIEVAL_LEARNING_RATE,
    RETRIEVAL_WEIGHT_DECAY,
    TRAIN_BATCH_SIZE,
    TRAIN_PATH,
    USER_MAPPING_PATH,
    VALIDATION_PATH,
)
from recsys.data.loader import (
    load_json,
    load_parquet,
    save_json,
)
from recsys.data.sampling import UniformNegativeSampler
from recsys.models.ncf import (
    GMF,
    MLPRecommender,
    NeuMF,
    initialize_neumf_from_pretrained,
)
from recsys.models.two_tower import TwoTowerRecommender
from recsys.training import (
    train_bpr_epoch,
    train_pointwise_epoch,
)
from recsys.utils.artifacts import (
    build_model_release_lineage,
    canonical_id_mapping_sha256,
    canonical_interaction_pairs_sha256,
    canonical_json_sha256,
    sha256_file,
    validate_model_data_identity,
    validate_processed_data_identity,
)
from recsys.utils.artifacts import (
    require_metadata_dict as _require_dict,
)
from recsys.utils.artifacts import (
    require_metadata_string as _require_string,
)
from recsys.utils.artifacts import (
    validate_processed_manifest_file_hash as _validate_manifest_file_hash,
)
from recsys.utils.helpers import (
    get_torch_device,
    set_seed,
)
from recsys.utils.model_io import (
    load_model_checkpoint,
    load_tensor,
    save_model_checkpoint,
    save_tensor,
)

MODEL_RELEASES_DIR = FINAL_NEUMF_MODEL_PATH.parent / "releases"

MODEL_RELEASE_POINTER_PATH = FINAL_NEUMF_MODEL_PATH.parent / "current_release.json"

MODEL_RELEASE_MANIFEST_NAME = "model_release_manifest.json"

FINAL_TWO_TOWER_EMBEDDING_METADATA_NAME = "final_two_tower_embeddings_metadata.json"

MODEL_RELEASE_MANIFEST_VERSION = 1
MODEL_RELEASE_POINTER_VERSION = 1


def load_and_validate_training_lineage(
    validation: pl.DataFrame,
    user_mapping: pl.DataFrame,
    item_mapping: pl.DataFrame,
) -> dict[str, str]:
    """
    Validate the processed-data manifest against the exact artifacts
    consumed by final training and return immutable lineage metadata.
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

    consumed_artifact_hashes = {
        "train": train_sha256,
        "validation": validation_sha256,
        "user_mapping": (user_mapping_file_sha256),
        "item_mapping": (item_mapping_file_sha256),
    }

    for (
        artifact_name,
        actual_hash,
    ) in consumed_artifact_hashes.items():
        identity_hash = _require_string(
            identity_artifact_hashes,
            artifact_name,
            "identity.artifact_hashes",
        )

        if identity_hash != actual_hash:
            raise RuntimeError(
                f"Processed-data identity does not match artifact '{artifact_name}'."
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

    identity_mapping_hashes = _require_dict(
        identity_payload.get("semantic_mapping_hashes"),
        "identity.semantic_mapping_hashes",
    )

    if (
        _require_string(
            identity_mapping_hashes,
            "user_mapping_sha256",
            "identity.semantic_mapping_hashes",
        )
        != expected_user_mapping_sha256
    ):
        raise RuntimeError(
            "User mapping fingerprint differs between manifest lineage sections."
        )

    if (
        _require_string(
            identity_mapping_hashes,
            "item_mapping_sha256",
            "identity.semantic_mapping_hashes",
        )
        != expected_item_mapping_sha256
    ):
        raise RuntimeError(
            "Item mapping fingerprint differs between manifest lineage sections."
        )

    manifest_preprocessing_config = _require_dict(
        manifest.get("preprocessing_config"),
        "preprocessing_config",
    )

    identity_preprocessing_config = _require_dict(
        identity_payload.get("preprocessing_config"),
        "identity.preprocessing_config",
    )

    if manifest_preprocessing_config != identity_preprocessing_config:
        raise RuntimeError(
            "Preprocessing configuration differs between manifest lineage sections."
        )

    if manifest_preprocessing_config.get("n_users") != user_mapping.height:
        raise RuntimeError(
            "Processed manifest user count does not match the current user mapping."
        )

    if manifest_preprocessing_config.get("n_items") != item_mapping.height:
        raise RuntimeError(
            "Processed manifest item count does not match the current item mapping."
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

    return build_model_release_lineage(model_data_identity)


def build_two_tower_embedding_metadata(
    *,
    checkpoint_path: Path,
    user_embeddings_path: Path,
    item_embeddings_path: Path,
    user_embeddings: torch.Tensor,
    item_embeddings: torch.Tensor,
    lineage: dict[str, str],
) -> dict:
    """
    Bind exported embeddings to the exact staged Two-Tower checkpoint
    and processed-data lineage that produced them.
    """

    checkpoint_sha256 = sha256_file(checkpoint_path)

    user_embeddings_sha256 = sha256_file(user_embeddings_path)

    item_embeddings_sha256 = sha256_file(item_embeddings_path)

    return {
        "metadata_version": 1,
        "model": "TwoTowerRecommender",
        "parent_checkpoint": {
            "filename": checkpoint_path.name,
            "sha256": checkpoint_sha256,
        },
        "user_embeddings": {
            "filename": user_embeddings_path.name,
            "sha256": user_embeddings_sha256,
            "shape": list(user_embeddings.shape),
        },
        "item_embeddings": {
            "filename": item_embeddings_path.name,
            "sha256": item_embeddings_sha256,
            "shape": list(item_embeddings.shape),
        },
        "lineage": dict(lineage),
    }


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
    data: dict,
    path: Path,
) -> None:
    """
    Atomically replace one JSON pointer file.

    The new pointer becomes visible only after the complete JSON payload
    has been written and fsynced in the destination directory.
    """

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


def _validate_release_checkpoint(
    checkpoint: dict,
    checkpoint_name: str,
    lineage: dict[str, str],
    n_users: int,
    n_items: int,
) -> None:
    """Validate a staged final checkpoint before release publication."""

    model_config = _require_dict(
        checkpoint.get("model_config"),
        f"{checkpoint_name}.model_config",
    )

    training_config = _require_dict(
        checkpoint.get("training_config"),
        f"{checkpoint_name}.training_config",
    )

    if model_config.get("n_users") != n_users:
        raise RuntimeError(f"{checkpoint_name} staged user vocabulary mismatch.")

    if model_config.get("n_items") != n_items:
        raise RuntimeError(f"{checkpoint_name} staged item vocabulary mismatch.")

    if training_config.get("training_data") != FINAL_TRAINING_PROTOCOL:
        raise RuntimeError(f"{checkpoint_name} staged training protocol mismatch.")

    checkpoint_lineage = _require_dict(
        training_config.get("lineage"),
        f"{checkpoint_name}.training_config.lineage",
    )

    if checkpoint_lineage != lineage:
        raise RuntimeError(f"{checkpoint_name} staged lineage mismatch.")


def validate_staged_model_release(
    *,
    staging_dir: Path,
    lineage: dict[str, str],
    n_users: int,
    n_items: int,
) -> tuple[
    str,
    str,
    dict,
]:
    """
    Validate a complete staged production-model generation.

    The returned model-release ID is content-derived from the exact
    staged checkpoint/embedding/metadata hashes.
    """

    neumf_path = staging_dir / FINAL_NEUMF_MODEL_PATH.name

    two_tower_path = staging_dir / FINAL_TWO_TOWER_MODEL_PATH.name

    user_embeddings_path = staging_dir / FINAL_TWO_TOWER_USER_EMBEDDINGS_PATH.name

    item_embeddings_path = staging_dir / FINAL_TWO_TOWER_ITEM_EMBEDDINGS_PATH.name

    embedding_metadata_path = staging_dir / FINAL_TWO_TOWER_EMBEDDING_METADATA_NAME

    required_paths = (
        neumf_path,
        two_tower_path,
        user_embeddings_path,
        item_embeddings_path,
        embedding_metadata_path,
    )

    for artifact_path in required_paths:
        if not artifact_path.is_file():
            raise FileNotFoundError(
                f"Staged model release is missing required artifact: {artifact_path}"
            )

    neumf_checkpoint = load_model_checkpoint(
        neumf_path,
        device="cpu",
    )

    two_tower_checkpoint = load_model_checkpoint(
        two_tower_path,
        device="cpu",
    )

    _validate_release_checkpoint(
        checkpoint=neumf_checkpoint,
        checkpoint_name="NeuMF",
        lineage=lineage,
        n_users=n_users,
        n_items=n_items,
    )

    _validate_release_checkpoint(
        checkpoint=two_tower_checkpoint,
        checkpoint_name="Two-Tower",
        lineage=lineage,
        n_users=n_users,
        n_items=n_items,
    )

    user_embeddings = load_tensor(
        user_embeddings_path,
        device="cpu",
    )

    item_embeddings = load_tensor(
        item_embeddings_path,
        device="cpu",
    )

    if user_embeddings.ndim != 2:
        raise RuntimeError("Staged Two-Tower user embeddings must be 2D.")

    if item_embeddings.ndim != 2:
        raise RuntimeError("Staged Two-Tower item embeddings must be 2D.")

    if user_embeddings.shape[0] != n_users:
        raise RuntimeError("Staged Two-Tower user embedding count mismatch.")

    if item_embeddings.shape[0] != n_items:
        raise RuntimeError("Staged Two-Tower item embedding count mismatch.")

    if user_embeddings.shape[1] != item_embeddings.shape[1]:
        raise RuntimeError("Staged Two-Tower embedding dimensions do not match.")

    if not torch.isfinite(user_embeddings).all():
        raise RuntimeError("Staged user embeddings contain non-finite values.")

    if not torch.isfinite(item_embeddings).all():
        raise RuntimeError("Staged item embeddings contain non-finite values.")

    embedding_metadata = load_json(embedding_metadata_path)

    if embedding_metadata.get("metadata_version") != 1:
        raise RuntimeError("Unsupported staged embedding metadata version.")

    metadata_lineage = _require_dict(
        embedding_metadata.get("lineage"),
        "embedding_metadata.lineage",
    )

    if metadata_lineage != lineage:
        raise RuntimeError("Staged embedding lineage mismatch.")

    parent_checkpoint = _require_dict(
        embedding_metadata.get("parent_checkpoint"),
        "embedding_metadata.parent_checkpoint",
    )

    if parent_checkpoint.get("sha256") != sha256_file(two_tower_path):
        raise RuntimeError(
            "Staged embeddings are not bound to the staged Two-Tower checkpoint."
        )

    user_metadata = _require_dict(
        embedding_metadata.get("user_embeddings"),
        "embedding_metadata.user_embeddings",
    )

    item_metadata = _require_dict(
        embedding_metadata.get("item_embeddings"),
        "embedding_metadata.item_embeddings",
    )

    if user_metadata.get("sha256") != sha256_file(user_embeddings_path):
        raise RuntimeError("Staged user embedding fingerprint mismatch.")

    if item_metadata.get("sha256") != sha256_file(item_embeddings_path):
        raise RuntimeError("Staged item embedding fingerprint mismatch.")

    if user_metadata.get("shape") != list(user_embeddings.shape):
        raise RuntimeError("Staged user embedding shape metadata mismatch.")

    if item_metadata.get("shape") != list(item_embeddings.shape):
        raise RuntimeError("Staged item embedding shape metadata mismatch.")

    artifact_paths = {
        "neumf_model": neumf_path,
        "two_tower_model": two_tower_path,
        "two_tower_user_embeddings": (user_embeddings_path),
        "two_tower_item_embeddings": (item_embeddings_path),
        "two_tower_embedding_metadata": (embedding_metadata_path),
    }

    artifact_hashes = {
        artifact_name: sha256_file(artifact_path)
        for (
            artifact_name,
            artifact_path,
        ) in sorted(artifact_paths.items())
    }

    release_identity = {
        "model_release_manifest_version": (MODEL_RELEASE_MANIFEST_VERSION),
        "lineage": dict(lineage),
        "artifact_hashes": (artifact_hashes),
    }

    release_fingerprint = canonical_json_sha256(release_identity)

    model_release_id = f"model-{release_fingerprint[:20]}"

    manifest = {
        "model_release_manifest_version": (MODEL_RELEASE_MANIFEST_VERSION),
        "model_release_id": (model_release_id),
        "model_release_fingerprint": (release_fingerprint),
        "lineage": dict(lineage),
        "catalog": {
            "n_users": n_users,
            "n_items": n_items,
        },
        "artifacts": {
            artifact_name: {
                "filename": artifact_path.name,
                "size_bytes": (artifact_path.stat().st_size),
                "sha256": (artifact_hashes[artifact_name]),
            }
            for (
                artifact_name,
                artifact_path,
            ) in artifact_paths.items()
        },
    }

    manifest_path = staging_dir / MODEL_RELEASE_MANIFEST_NAME

    save_json(
        manifest,
        manifest_path,
    )

    manifest_sha256 = sha256_file(manifest_path)

    return (
        model_release_id,
        manifest_sha256,
        manifest,
    )


def publish_staged_model_release(
    *,
    staging_dir: Path,
    model_release_id: str,
    manifest_sha256: str,
    lineage: dict[str, str],
) -> Path:
    """
    Publish an immutable model generation, then atomically switch
    the single current-model pointer to it.
    """

    MODEL_RELEASES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    final_release_dir = MODEL_RELEASES_DIR / model_release_id

    if final_release_dir.exists():
        existing_manifest_path = final_release_dir / MODEL_RELEASE_MANIFEST_NAME

        if (
            not existing_manifest_path.is_file()
            or sha256_file(existing_manifest_path) != manifest_sha256
        ):
            raise RuntimeError(
                "A model release directory with the same "
                "content-derived ID already exists but does "
                "not match the staged release."
            )

        shutil.rmtree(staging_dir)

    else:
        os.replace(
            staging_dir,
            final_release_dir,
        )

        _fsync_directory(MODEL_RELEASES_DIR)

    pointer = {
        "pointer_version": (MODEL_RELEASE_POINTER_VERSION),
        "model_release_id": (model_release_id),
        "model_release_dir": (f"releases/{model_release_id}"),
        "manifest_filename": (MODEL_RELEASE_MANIFEST_NAME),
        "manifest_sha256": (manifest_sha256),
        "release_lineage_id": (lineage["release_lineage_id"]),
        "release_fingerprint": (lineage["release_fingerprint"]),
        "data_run_id": (lineage["data_run_id"]),
        "dataset_fingerprint": (lineage["dataset_fingerprint"]),
    }

    _atomic_save_json(
        pointer,
        MODEL_RELEASE_POINTER_PATH,
    )

    return final_release_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train final production recommendation models "
            "on training interactions plus warm validation interactions."
        )
    )

    parser.add_argument(
        "--models",
        choices=("all",),
        default="all",
        help="Train and atomically publish both production models and embeddings.",
    )

    return parser.parse_args()


def clear_device_cache(
    device: torch.device,
) -> None:
    """Release Python and accelerator caches."""

    gc.collect()

    if device.type == "mps":
        torch.mps.empty_cache()

    elif device.type == "cuda":
        torch.cuda.empty_cache()


def build_training_arrays(
    interactions: pl.DataFrame,
    n_users: int,
    n_items: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    """
    Convert encoded interactions to validated NumPy arrays.

    Final model training accepts only known non-negative
    user and item indices.
    """

    required_columns = {
        "user_idx",
        "item_idx",
    }

    missing_columns = required_columns - set(interactions.columns)

    if missing_columns:
        raise ValueError(
            f"Training interactions are missing columns: {sorted(missing_columns)}"
        )

    if interactions.is_empty():
        raise ValueError("Training interactions cannot be empty.")

    user_ids = (
        interactions["user_idx"]
        .to_numpy()
        .astype(
            np.int64,
            copy=False,
        )
    )

    item_ids = (
        interactions["item_idx"]
        .to_numpy()
        .astype(
            np.int64,
            copy=False,
        )
    )

    if np.any(user_ids < 0):
        raise ValueError("Final training data contains negative user indices.")

    if np.any(user_ids >= n_users):
        raise ValueError(
            "Final training data contains user indices outside the user vocabulary."
        )

    if np.any(item_ids < 0):
        raise ValueError("Final training data contains negative item indices.")

    if np.any(item_ids >= n_items):
        raise ValueError(
            "Final training data contains item indices outside the item vocabulary."
        )

    return (
        user_ids,
        item_ids,
    )


def prepare_final_training_interactions(
    train: pl.DataFrame,
    validation: pl.DataFrame,
    n_users: int,
    n_items: int,
) -> tuple[
    pl.DataFrame,
    int,
    int,
]:
    """
    Build the final training set.

    The item vocabulary remains frozen from the original
    training partition.

    Warm validation interactions are added to final training.
    Cold validation items encoded as -1 are excluded.
    """

    required_columns = {
        "user_idx",
        "item_idx",
    }

    for frame_name, frame in (
        ("train", train),
        ("validation", validation),
    ):
        missing_columns = required_columns - set(frame.columns)

        if missing_columns:
            raise ValueError(
                f"{frame_name} is missing columns: {sorted(missing_columns)}"
            )

        if frame.is_empty():
            raise ValueError(f"{frame_name} cannot be empty.")

        if frame["user_idx"].null_count() > 0:
            raise ValueError(f"{frame_name} contains null user indices.")

        if frame["item_idx"].null_count() > 0:
            raise ValueError(f"{frame_name} contains null item indices.")

    if train.filter(pl.col("user_idx") < 0).height > 0:
        raise ValueError("Training data contains negative user indices.")

    if train.filter(pl.col("item_idx") < 0).height > 0:
        raise ValueError("Training data contains negative item indices.")

    if validation.filter(pl.col("user_idx") < 0).height > 0:
        raise ValueError("Validation contains unknown users.")

    if validation.filter(pl.col("item_idx") < COLD_ITEM_INDEX).height > 0:
        raise ValueError(
            f"Validation contains invalid item indices below {COLD_ITEM_INDEX}."
        )

    if train.filter(pl.col("user_idx") >= n_users).height > 0:
        raise ValueError("Training contains user indices outside the user vocabulary.")

    if validation.filter(pl.col("user_idx") >= n_users).height > 0:
        raise ValueError(
            "Validation contains user indices outside the user vocabulary."
        )

    if train.filter(pl.col("item_idx") >= n_items).height > 0:
        raise ValueError("Training contains item indices outside the item vocabulary.")

    if validation.filter(pl.col("item_idx") >= n_items).height > 0:
        raise ValueError(
            "Validation contains item indices outside the item vocabulary."
        )

    cold_validation_interactions = validation.filter(
        pl.col("item_idx") == COLD_ITEM_INDEX
    ).height

    warm_validation = validation.filter(pl.col("item_idx") >= 0)

    warm_validation_interactions = warm_validation.height

    interactions = pl.concat(
        [
            train,
            warm_validation,
        ],
        how="vertical",
    )

    if interactions.is_empty():
        raise RuntimeError("Final training interactions are empty.")

    if interactions.filter(pl.col("item_idx") < 0).height > 0:
        raise RuntimeError("Cold-item interactions entered final model training.")

    return (
        interactions,
        warm_validation_interactions,
        cold_validation_interactions,
    )


def create_negative_sampler(
    user_ids: np.ndarray,
    item_ids: np.ndarray,
    n_items: int,
    seed: int,
) -> UniformNegativeSampler:
    return UniformNegativeSampler(
        user_ids=user_ids,
        item_ids=item_ids,
        n_items=n_items,
        seed=seed,
    )


def export_two_tower_embeddings(
    model: TwoTowerRecommender,
    device: torch.device,
    n_users: int,
    n_items: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
]:
    """
    Export complete Two-Tower user/item embedding matrices.

    Embeddings are generated on the requested model device,
    validated against the expected vocabularies, detached,
    and returned on CPU for persistence.
    """

    model.eval()

    with torch.inference_mode():
        user_embeddings = model.get_user_embeddings(device=device).detach().cpu()

        item_embeddings = model.get_item_embeddings(device=device).detach().cpu()

    if user_embeddings.ndim != 2:
        raise RuntimeError("Two-Tower user embeddings must be a 2D tensor.")

    if item_embeddings.ndim != 2:
        raise RuntimeError("Two-Tower item embeddings must be a 2D tensor.")

    if user_embeddings.shape[0] != n_users:
        raise RuntimeError(
            "Two-Tower user embedding count does not match "
            f"the user vocabulary: "
            f"{user_embeddings.shape[0]} != {n_users}."
        )

    if item_embeddings.shape[0] != n_items:
        raise RuntimeError(
            "Two-Tower item embedding count does not match "
            f"the item vocabulary: "
            f"{item_embeddings.shape[0]} != {n_items}."
        )

    if user_embeddings.shape[1] != item_embeddings.shape[1]:
        raise RuntimeError("Two-Tower user/item embedding dimensions do not match.")

    return (
        user_embeddings,
        item_embeddings,
    )


def train_gmf(
    user_ids: np.ndarray,
    item_ids: np.ndarray,
    n_users: int,
    n_items: int,
    device: torch.device,
) -> tuple[
    GMF,
    list[float],
]:
    print("\n=== Training GMF ===")

    model = GMF(
        n_users=n_users,
        n_items=n_items,
        embedding_dim=(NCF_EMBEDDING_DIM),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=NCF_LEARNING_RATE,
        weight_decay=(NCF_WEIGHT_DECAY),
    )

    negative_sampler = create_negative_sampler(
        user_ids=user_ids,
        item_ids=item_ids,
        n_items=n_items,
        seed=RANDOM_SEED,
    )

    losses: list[float] = []

    for epoch in range(
        1,
        NCF_EPOCHS + 1,
    ):
        loss = train_pointwise_epoch(
            model=model,
            optimizer=optimizer,
            user_ids=user_ids,
            item_ids=item_ids,
            negative_sampler=(negative_sampler),
            batch_size=(TRAIN_BATCH_SIZE),
            device=device,
            seed=(RANDOM_SEED + epoch),
            desc=(f"GMF {epoch}/{NCF_EPOCHS}"),
        )

        losses.append(loss)

        print(f"GMF epoch {epoch:02d}/{NCF_EPOCHS}: loss={loss:.3f}")

    return (
        model,
        losses,
    )


def train_mlp(
    user_ids: np.ndarray,
    item_ids: np.ndarray,
    n_users: int,
    n_items: int,
    device: torch.device,
) -> tuple[
    MLPRecommender,
    list[float],
]:
    print("\n=== Training MLP ===")

    model = MLPRecommender(
        n_users=n_users,
        n_items=n_items,
        embedding_dim=(NCF_EMBEDDING_DIM),
        hidden_layers=(NCF_MLP_LAYERS),
        item_score_batch_size=(NCF_ITEM_SCORE_BATCH_SIZE),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=NCF_LEARNING_RATE,
        weight_decay=(NCF_WEIGHT_DECAY),
    )

    negative_sampler = create_negative_sampler(
        user_ids=user_ids,
        item_ids=item_ids,
        n_items=n_items,
        seed=(RANDOM_SEED + 1_000),
    )

    losses: list[float] = []

    for epoch in range(
        1,
        NCF_EPOCHS + 1,
    ):
        loss = train_pointwise_epoch(
            model=model,
            optimizer=optimizer,
            user_ids=user_ids,
            item_ids=item_ids,
            negative_sampler=(negative_sampler),
            batch_size=(TRAIN_BATCH_SIZE),
            device=device,
            seed=(RANDOM_SEED + 1_000 + epoch),
            desc=(f"MLP {epoch}/{NCF_EPOCHS}"),
        )

        losses.append(loss)

        print(f"MLP epoch {epoch:02d}/{NCF_EPOCHS}: loss={loss:.3f}")

    return (
        model,
        losses,
    )


def train_neumf(
    user_ids: np.ndarray,
    item_ids: np.ndarray,
    n_users: int,
    n_items: int,
    device: torch.device,
    final_training_interactions: int,
    warm_validation_interactions: int,
    excluded_cold_validation_interactions: int,
    lineage: dict[str, str],
    output_path: Path,
) -> None:
    (
        gmf,
        gmf_losses,
    ) = train_gmf(
        user_ids=user_ids,
        item_ids=item_ids,
        n_users=n_users,
        n_items=n_items,
        device=device,
    )

    (
        mlp,
        mlp_losses,
    ) = train_mlp(
        user_ids=user_ids,
        item_ids=item_ids,
        n_users=n_users,
        n_items=n_items,
        device=device,
    )

    print("\n=== Initializing NeuMF ===")

    model = NeuMF(
        n_users=n_users,
        n_items=n_items,
        embedding_dim=(NCF_EMBEDDING_DIM),
        hidden_layers=(NCF_MLP_LAYERS),
        item_score_batch_size=(NCF_ITEM_SCORE_BATCH_SIZE),
    ).to(device)

    initialize_neumf_from_pretrained(
        neumf=model,
        gmf=gmf,
        mlp=mlp,
        alpha=(NEUMF_PRETRAIN_ALPHA),
    )

    del gmf
    del mlp

    clear_device_cache(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=NEUMF_LEARNING_RATE,
        weight_decay=(NEUMF_WEIGHT_DECAY),
    )

    negative_sampler = create_negative_sampler(
        user_ids=user_ids,
        item_ids=item_ids,
        n_items=n_items,
        seed=(RANDOM_SEED + 2_000),
    )

    neumf_losses: list[float] = []

    print("\n=== Training NeuMF ===")

    for epoch in range(
        1,
        NEUMF_EPOCHS + 1,
    ):
        loss = train_pointwise_epoch(
            model=model,
            optimizer=optimizer,
            user_ids=user_ids,
            item_ids=item_ids,
            negative_sampler=(negative_sampler),
            batch_size=(TRAIN_BATCH_SIZE),
            device=device,
            seed=(RANDOM_SEED + 2_000 + epoch),
            desc=(f"NeuMF {epoch}/{NEUMF_EPOCHS}"),
        )

        neumf_losses.append(loss)

        print(f"NeuMF epoch {epoch:02d}/{NEUMF_EPOCHS}: loss={loss:.3f}")

    model.eval()

    save_model_checkpoint(
        model=model,
        path=output_path,
        model_config={
            "n_users": n_users,
            "n_items": n_items,
            "embedding_dim": (NCF_EMBEDDING_DIM),
            "hidden_layers": list(NCF_MLP_LAYERS),
            "item_score_batch_size": (NCF_ITEM_SCORE_BATCH_SIZE),
        },
        metrics={
            "final_training_loss": (neumf_losses[-1]),
        },
        training_config={
            "feedback": "implicit",
            "training_data": (FINAL_TRAINING_PROTOCOL),
            "lineage": dict(lineage),
            "final_training_interactions": (final_training_interactions),
            "warm_validation_interactions": (warm_validation_interactions),
            "excluded_cold_validation_interactions": (
                excluded_cold_validation_interactions
            ),
            "cold_item_index": (COLD_ITEM_INDEX),
            "optimizer": "AdamW",
            "learning_rate": (NEUMF_LEARNING_RATE),
            "weight_decay": (NEUMF_WEIGHT_DECAY),
            "epochs": (NEUMF_EPOCHS),
            "batch_size": (TRAIN_BATCH_SIZE),
            "pretrain_alpha": (NEUMF_PRETRAIN_ALPHA),
            "gmf_loss_history": (gmf_losses),
            "mlp_loss_history": (mlp_losses),
            "neumf_loss_history": (neumf_losses),
            "seed": (RANDOM_SEED),
        },
    )

    print(f"\nStaged NeuMF:\n{output_path}")

    del model

    clear_device_cache(device)


def train_two_tower(
    user_ids: np.ndarray,
    item_ids: np.ndarray,
    n_users: int,
    n_items: int,
    device: torch.device,
    final_training_interactions: int,
    warm_validation_interactions: int,
    excluded_cold_validation_interactions: int,
    lineage: dict[str, str],
    checkpoint_path: Path,
    user_embeddings_path: Path,
    item_embeddings_path: Path,
    embedding_metadata_path: Path,
) -> None:
    print("\n=== Training Two-Tower ===")

    model = TwoTowerRecommender(
        n_users=n_users,
        n_items=n_items,
        embedding_dim=(RETRIEVAL_EMBEDDING_DIM),
        hidden_dim=(RETRIEVAL_HIDDEN_DIM),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=(RETRIEVAL_LEARNING_RATE),
        weight_decay=(RETRIEVAL_WEIGHT_DECAY),
    )

    negative_sampler = create_negative_sampler(
        user_ids=user_ids,
        item_ids=item_ids,
        n_items=n_items,
        seed=(RANDOM_SEED + 3_000),
    )

    losses: list[float] = []

    for epoch in range(
        1,
        RETRIEVAL_EPOCHS + 1,
    ):
        loss = train_bpr_epoch(
            model=model,
            optimizer=optimizer,
            user_ids=user_ids,
            item_ids=item_ids,
            negative_sampler=(negative_sampler),
            batch_size=(TRAIN_BATCH_SIZE),
            device=device,
            seed=(RANDOM_SEED + 3_000 + epoch),
            desc=(f"Two-Tower {epoch}/{RETRIEVAL_EPOCHS}"),
        )

        losses.append(loss)

        print(f"Two-Tower epoch {epoch:02d}/{RETRIEVAL_EPOCHS}: loss={loss:.3f}")

    model.eval()

    # Export first so a broken embedding-export path cannot
    # leave behind a newly written checkpoint paired with
    # stale embedding files.
    print("\nGenerating final retrieval embeddings...")

    (
        user_embeddings,
        item_embeddings,
    ) = export_two_tower_embeddings(
        model=model,
        device=device,
        n_users=n_users,
        n_items=n_items,
    )

    save_model_checkpoint(
        model=model,
        path=checkpoint_path,
        model_config={
            "n_users": n_users,
            "n_items": n_items,
            "embedding_dim": (RETRIEVAL_EMBEDDING_DIM),
            "hidden_dim": (RETRIEVAL_HIDDEN_DIM),
        },
        metrics={
            "final_training_loss": (losses[-1]),
        },
        training_config={
            "feedback": "implicit",
            "objective": "BPR",
            "training_data": (FINAL_TRAINING_PROTOCOL),
            "lineage": dict(lineage),
            "final_training_interactions": (final_training_interactions),
            "warm_validation_interactions": (warm_validation_interactions),
            "excluded_cold_validation_interactions": (
                excluded_cold_validation_interactions
            ),
            "cold_item_index": (COLD_ITEM_INDEX),
            "optimizer": "AdamW",
            "learning_rate": (RETRIEVAL_LEARNING_RATE),
            "weight_decay": (RETRIEVAL_WEIGHT_DECAY),
            "epochs": (RETRIEVAL_EPOCHS),
            "batch_size": (TRAIN_BATCH_SIZE),
            "loss_history": (losses),
            "seed": (RANDOM_SEED),
        },
    )

    save_tensor(
        user_embeddings,
        user_embeddings_path,
    )

    save_tensor(
        item_embeddings,
        item_embeddings_path,
    )

    embedding_metadata = build_two_tower_embedding_metadata(
        checkpoint_path=checkpoint_path,
        user_embeddings_path=user_embeddings_path,
        item_embeddings_path=item_embeddings_path,
        user_embeddings=user_embeddings,
        item_embeddings=item_embeddings,
        lineage=lineage,
    )

    save_json(
        embedding_metadata,
        embedding_metadata_path,
    )

    print(f"\nStaged Two-Tower:\n{checkpoint_path}")

    print(f"\nStaged user embeddings:\n{user_embeddings_path}")

    print(f"\nStaged item embeddings:\n{item_embeddings_path}")

    print(f"\nStaged embedding metadata:\n{embedding_metadata_path}")

    print(
        "\nEmbedding shapes:"
        f"\nusers = {tuple(user_embeddings.shape)}"
        f"\nitems = {tuple(item_embeddings.shape)}"
    )

    del model
    del user_embeddings
    del item_embeddings

    clear_device_cache(device)


def main() -> None:
    args = parse_args()

    if args.models != "all":
        raise ValueError(
            "Atomic production publication requires --models all. "
            "A complete release must contain NeuMF, Two-Tower, "
            "and both Two-Tower embedding matrices."
        )

    started = time.perf_counter()

    set_seed(RANDOM_SEED)

    device = get_torch_device()

    print(f"Device: {device}")

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

    lineage = load_and_validate_training_lineage(
        validation=validation,
        user_mapping=user_mapping,
        item_mapping=item_mapping,
    )

    n_users = user_mapping.height
    n_items = item_mapping.height

    if n_users <= 0:
        raise RuntimeError("User mapping is empty.")

    if n_items <= 0:
        raise RuntimeError("Item mapping is empty.")

    (
        interactions,
        warm_validation_interactions,
        cold_validation_interactions,
    ) = prepare_final_training_interactions(
        train=train,
        validation=validation,
        n_users=n_users,
        n_items=n_items,
    )

    print("\nFinal training data:")

    print(f"Interactions            : {interactions.height:,}")

    print(f"Training interactions   : {train.height:,}")

    print(f"Warm validation added   : {warm_validation_interactions:,}")

    print(f"Cold validation excluded: {cold_validation_interactions:,}")

    print(f"Users                   : {n_users:,}")

    print(f"Items                   : {n_items:,}")

    print(f"Data run ID             : {lineage['data_run_id']}")

    print(f"Release lineage ID      : {lineage['release_lineage_id']}")

    (
        user_ids,
        item_ids,
    ) = build_training_arrays(
        interactions=interactions,
        n_users=n_users,
        n_items=n_items,
    )

    final_training_interactions = interactions.height

    del train
    del validation
    del interactions
    del user_mapping
    del item_mapping

    gc.collect()

    MODEL_RELEASES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=".staging-",
            dir=MODEL_RELEASES_DIR,
        )
    )

    try:
        staged_neumf_path = staging_dir / FINAL_NEUMF_MODEL_PATH.name

        staged_two_tower_path = staging_dir / FINAL_TWO_TOWER_MODEL_PATH.name

        staged_user_embeddings_path = (
            staging_dir / FINAL_TWO_TOWER_USER_EMBEDDINGS_PATH.name
        )

        staged_item_embeddings_path = (
            staging_dir / FINAL_TWO_TOWER_ITEM_EMBEDDINGS_PATH.name
        )

        staged_embedding_metadata_path = (
            staging_dir / FINAL_TWO_TOWER_EMBEDDING_METADATA_NAME
        )

        train_neumf(
            user_ids=user_ids,
            item_ids=item_ids,
            n_users=n_users,
            n_items=n_items,
            device=device,
            final_training_interactions=(final_training_interactions),
            warm_validation_interactions=(warm_validation_interactions),
            excluded_cold_validation_interactions=(cold_validation_interactions),
            lineage=lineage,
            output_path=staged_neumf_path,
        )

        train_two_tower(
            user_ids=user_ids,
            item_ids=item_ids,
            n_users=n_users,
            n_items=n_items,
            device=device,
            final_training_interactions=(final_training_interactions),
            warm_validation_interactions=(warm_validation_interactions),
            excluded_cold_validation_interactions=(cold_validation_interactions),
            lineage=lineage,
            checkpoint_path=(staged_two_tower_path),
            user_embeddings_path=(staged_user_embeddings_path),
            item_embeddings_path=(staged_item_embeddings_path),
            embedding_metadata_path=(staged_embedding_metadata_path),
        )

        print("\nValidating staged model release...")

        (
            model_release_id,
            manifest_sha256,
            _,
        ) = validate_staged_model_release(
            staging_dir=staging_dir,
            lineage=lineage,
            n_users=n_users,
            n_items=n_items,
        )

        final_release_dir = publish_staged_model_release(
            staging_dir=staging_dir,
            model_release_id=(model_release_id),
            manifest_sha256=(manifest_sha256),
            lineage=lineage,
        )

    except Exception:
        if staging_dir.exists():
            shutil.rmtree(
                staging_dir,
                ignore_errors=True,
            )

        raise

    elapsed = time.perf_counter() - started

    print("\nTraining complete.")

    print(f"Published model release : {model_release_id}")

    print(f"Release directory       : {final_release_dir}")

    print(f"Current release pointer : {MODEL_RELEASE_POINTER_PATH}")

    print(f"Elapsed: {elapsed / 60:.3f} minutes")


if __name__ == "__main__":
    main()
