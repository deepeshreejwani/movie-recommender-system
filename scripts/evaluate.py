from __future__ import annotations

import json
import time
from typing import Any

import numpy as np
import polars as pl
import torch

from recsys.config import (
    ARTIFACT_MANIFEST_PATH,
    COLD_ITEM_INDEX,
    EVAL_BATCH_SIZE,
    EVAL_KS,
    FINAL_TRAINING_PROTOCOL,
    ITEM_MAPPING_PATH,
    PREPROCESSING_PROTOCOL_VERSION,
    PROCESSED_ARTIFACT_MANIFEST_VERSION,
    RANDOM_SEED,
    RANKING_CANDIDATE_SIZE,
    RANKING_RESULTS_PATH,
    RANKING_SCORE_BATCH_SIZE,
    TEST_PATH,
    TRAIN_PATH,
    USER_MAPPING_PATH,
    VALIDATION_PATH,
)
from recsys.data.loader import (
    load_json,
    load_parquet,
    save_json,
)
from recsys.evaluation.evaluator import (
    build_seen_items_csr,
    build_target_array,
    evaluate_embedding_retrieval,
    evaluate_retrieval_reranking,
)
from recsys.models.ncf import NeuMF
from recsys.models.two_tower import TwoTowerRecommender
from recsys.utils.artifacts import (
    build_model_release_lineage,
    canonical_id_mapping_sha256,
    canonical_interaction_pairs_sha256,
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
    CHECKPOINT_VERSION,
    load_model_checkpoint,
    load_tensor,
    restore_model_from_checkpoint,
)
from recsys.utils.releases import (
    ModelRelease,
    resolve_current_model_release,
)


def rounded(
    value: object,
) -> object:
    """Recursively round floating-point values for console display."""

    if isinstance(value, float):
        return round(value, 3)

    if isinstance(value, dict):
        return {key: rounded(item) for key, item in value.items()}

    if isinstance(value, list):
        return [rounded(item) for item in value]

    if isinstance(value, tuple):
        return [rounded(item) for item in value]

    return value


def load_and_validate_evaluation_lineage(
    validation: pl.DataFrame,
    user_mapping: pl.DataFrame,
    item_mapping: pl.DataFrame,
) -> tuple[
    dict[str, str],
    dict[str, str],
]:
    """
    Validate the processed-data manifest against the exact artifacts
    consumed by final evaluation.

    Model lineage is intentionally derived from train, validation, and
    the frozen mappings because the final test identity must never affect
    model training. Test SHA is returned separately as evaluation lineage.
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

    artifact_paths = {
        "train": TRAIN_PATH,
        "validation": VALIDATION_PATH,
        "test": TEST_PATH,
        "user_mapping": USER_MAPPING_PATH,
        "item_mapping": ITEM_MAPPING_PATH,
    }

    actual_artifact_hashes = {
        artifact_name: _validate_manifest_file_hash(
            processed_artifacts=processed_artifacts,
            artifact_name=artifact_name,
            path=artifact_path,
        )
        for artifact_name, artifact_path in artifact_paths.items()
    }

    for artifact_name, actual_hash in actual_artifact_hashes.items():
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
        "train_sha256": actual_artifact_hashes["train"],
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

    model_lineage = build_model_release_lineage(model_data_identity)

    evaluation_lineage = {
        "data_run_id": _require_string(
            manifest,
            "data_run_id",
            "artifact_manifest",
        ),
        "dataset_fingerprint": _require_string(
            manifest,
            "dataset_fingerprint",
            "artifact_manifest",
        ),
        "test_sha256": actual_artifact_hashes["test"],
    }

    return (
        model_lineage,
        evaluation_lineage,
    )


def validate_encoded_frame(
    frame: pl.DataFrame,
    frame_name: str,
    n_users: int,
    n_items: int,
    allow_cold_items: bool,
) -> None:
    """
    Validate encoded interaction bounds.

    Training-style frames must contain only known items.
    Holdout frames may contain item_idx == COLD_ITEM_INDEX.
    """

    required_columns = {
        "user_idx",
        "item_idx",
    }

    missing_columns = required_columns - set(frame.columns)

    if missing_columns:
        raise ValueError(f"{frame_name} is missing columns: {sorted(missing_columns)}")

    if frame.is_empty():
        raise ValueError(f"{frame_name} cannot be empty.")

    if frame["user_idx"].null_count() > 0:
        raise ValueError(f"{frame_name} contains null user indices.")

    if frame["item_idx"].null_count() > 0:
        raise ValueError(f"{frame_name} contains null item indices.")

    if frame.filter(pl.col("user_idx") < 0).height > 0:
        raise ValueError(f"{frame_name} contains negative user indices.")

    if frame.filter(pl.col("user_idx") >= n_users).height > 0:
        raise ValueError(
            f"{frame_name} contains user indices outside the user vocabulary."
        )

    minimum_item_index = COLD_ITEM_INDEX if allow_cold_items else 0

    if frame.filter(pl.col("item_idx") < minimum_item_index).height > 0:
        raise ValueError(
            f"{frame_name} contains item indices below {minimum_item_index}."
        )

    if frame.filter(pl.col("item_idx") >= n_items).height > 0:
        raise ValueError(
            f"{frame_name} contains item indices outside the item vocabulary."
        )


def prepare_seen_interactions(
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
    Build the interaction history that was available to final models.

    Final models are trained using:

        train + warm validation

    Cold validation items are outside the frozen training catalog and
    therefore cannot be added to final training or seen-item history.
    """

    validate_encoded_frame(
        frame=train,
        frame_name="train",
        n_users=n_users,
        n_items=n_items,
        allow_cold_items=False,
    )

    validate_encoded_frame(
        frame=validation,
        frame_name="validation",
        n_users=n_users,
        n_items=n_items,
        allow_cold_items=True,
    )

    cold_validation_count = validation.filter(
        pl.col("item_idx") == COLD_ITEM_INDEX
    ).height

    warm_validation = validation.filter(pl.col("item_idx") >= 0)

    warm_validation_count = warm_validation.height

    seen_interactions = pl.concat(
        [
            train,
            warm_validation,
        ],
        how="vertical",
    )

    validate_encoded_frame(
        frame=seen_interactions,
        frame_name="seen_interactions",
        n_users=n_users,
        n_items=n_items,
        allow_cold_items=False,
    )

    return (
        seen_interactions,
        warm_validation_count,
        cold_validation_count,
    )


def validate_test_targets(
    test: pl.DataFrame,
    n_users: int,
    n_items: int,
) -> None:
    """
    Validate final test holdout structure.

    Exactly one target must exist per user.

    Cold targets encoded as -1 are valid and remain part
    of the final evaluation population.
    """

    validate_encoded_frame(
        frame=test,
        frame_name="test",
        n_users=n_users,
        n_items=n_items,
        allow_cold_items=True,
    )

    if test.height != n_users:
        raise ValueError(
            "Test must contain exactly one "
            "interaction per evaluation user. "
            f"rows={test.height}, users={n_users}."
        )

    if test["user_idx"].n_unique() != n_users:
        raise ValueError("Test must contain exactly one unique target per user.")


def validate_final_checkpoint(
    checkpoint: dict[str, Any],
    checkpoint_name: str,
    expected_lineage: dict[str, str],
    n_users: int,
    n_items: int,
    final_training_interactions: int,
    warm_validation_interactions: int,
    excluded_cold_validation_interactions: int,
) -> None:
    """Validate final-model protocol, counts, vocabulary, and lineage."""

    checkpoint_version = checkpoint.get("checkpoint_version")

    if checkpoint_version != CHECKPOINT_VERSION:
        raise RuntimeError(
            f"{checkpoint_name} checkpoint version mismatch: "
            f"expected={CHECKPOINT_VERSION}, actual={checkpoint_version}."
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
            f"{checkpoint_name} user vocabulary does not match evaluation data."
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
            f"{checkpoint_name} item vocabulary does not match evaluation data."
        )

    training_data = _require_string(
        training_config,
        "training_data",
        f"{checkpoint_name}.training_config",
    )

    if training_data != FINAL_TRAINING_PROTOCOL:
        raise RuntimeError(
            f"{checkpoint_name} training protocol mismatch: "
            f"expected={FINAL_TRAINING_PROTOCOL!r}, actual={training_data!r}."
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
            "does not match evaluation data."
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
            f"{checkpoint_name} warm-validation count does not match evaluation data."
        )

    if (
        _require_int(
            training_config,
            "excluded_cold_validation_interactions",
            f"{checkpoint_name}.training_config",
        )
        != excluded_cold_validation_interactions
    ):
        raise RuntimeError(
            f"{checkpoint_name} excluded cold-validation count "
            "does not match evaluation data."
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
            f"{checkpoint_name} lineage does not match the processed dataset. "
            f"Mismatched fields: {mismatched_keys}"
        )


def load_two_tower(
    device: torch.device,
    model_release: ModelRelease,
    expected_lineage: dict[str, str],
    n_users: int,
    n_items: int,
    final_training_interactions: int,
    warm_validation_interactions: int,
    excluded_cold_validation_interactions: int,
) -> tuple[
    TwoTowerRecommender,
    dict[str, Any],
    str,
]:
    checkpoint = load_model_checkpoint(
        path=model_release.two_tower_model_path,
        device="cpu",
    )

    validate_final_checkpoint(
        checkpoint=checkpoint,
        checkpoint_name="Two-Tower",
        expected_lineage=expected_lineage,
        n_users=n_users,
        n_items=n_items,
        final_training_interactions=final_training_interactions,
        warm_validation_interactions=warm_validation_interactions,
        excluded_cold_validation_interactions=(excluded_cold_validation_interactions),
    )

    config = _require_dict(
        checkpoint["model_config"],
        "Two-Tower.model_config",
    )

    model = TwoTowerRecommender(
        n_users=_require_int(
            config,
            "n_users",
            "Two-Tower.model_config",
        ),
        n_items=_require_int(
            config,
            "n_items",
            "Two-Tower.model_config",
        ),
        embedding_dim=_require_int(
            config,
            "embedding_dim",
            "Two-Tower.model_config",
        ),
        hidden_dim=_require_int(
            config,
            "hidden_dim",
            "Two-Tower.model_config",
        ),
    )

    model = restore_model_from_checkpoint(
        model=model,
        checkpoint=checkpoint,
        device=device,
    )

    checkpoint_sha256 = sha256_file(model_release.two_tower_model_path)

    return (
        model,
        checkpoint,
        checkpoint_sha256,
    )


def load_neumf(
    device: torch.device,
    model_release: ModelRelease,
    expected_lineage: dict[str, str],
    n_users: int,
    n_items: int,
    final_training_interactions: int,
    warm_validation_interactions: int,
    excluded_cold_validation_interactions: int,
) -> tuple[
    NeuMF,
    dict[str, Any],
    str,
]:
    checkpoint = load_model_checkpoint(
        path=model_release.neumf_model_path,
        device="cpu",
    )

    validate_final_checkpoint(
        checkpoint=checkpoint,
        checkpoint_name="NeuMF",
        expected_lineage=expected_lineage,
        n_users=n_users,
        n_items=n_items,
        final_training_interactions=final_training_interactions,
        warm_validation_interactions=warm_validation_interactions,
        excluded_cold_validation_interactions=(excluded_cold_validation_interactions),
    )

    config = _require_dict(
        checkpoint["model_config"],
        "NeuMF.model_config",
    )

    hidden_layers = config.get("hidden_layers")

    if not isinstance(hidden_layers, list) or not all(
        isinstance(value, int) for value in hidden_layers
    ):
        raise TypeError(
            "NeuMF.model_config['hidden_layers'] must be a list of integers."
        )

    model = NeuMF(
        n_users=_require_int(
            config,
            "n_users",
            "NeuMF.model_config",
        ),
        n_items=_require_int(
            config,
            "n_items",
            "NeuMF.model_config",
        ),
        embedding_dim=_require_int(
            config,
            "embedding_dim",
            "NeuMF.model_config",
        ),
        hidden_layers=tuple(hidden_layers),
        item_score_batch_size=_require_int(
            config,
            "item_score_batch_size",
            "NeuMF.model_config",
        ),
    )

    model = restore_model_from_checkpoint(
        model=model,
        checkpoint=checkpoint,
        device=device,
    )

    checkpoint_sha256 = sha256_file(model_release.neumf_model_path)

    return (
        model,
        checkpoint,
        checkpoint_sha256,
    )


def load_and_validate_two_tower_item_embeddings(
    *,
    device: torch.device,
    model_release: ModelRelease,
    two_tower_checkpoint: dict[str, Any],
    two_tower_checkpoint_sha256: str,
    expected_lineage: dict[str, str],
    n_items: int,
) -> tuple[
    torch.Tensor,
    str,
]:
    """Validate item embeddings against their parent checkpoint and lineage."""

    metadata = load_json(model_release.embedding_metadata_path)

    if metadata.get("metadata_version") != 1:
        raise RuntimeError("Unsupported Two-Tower embedding metadata version.")

    if metadata.get("model") != "TwoTowerRecommender":
        raise RuntimeError(
            "Two-Tower embedding metadata contains an unexpected model type."
        )

    parent_checkpoint = _require_dict(
        metadata.get("parent_checkpoint"),
        "embedding_metadata.parent_checkpoint",
    )

    expected_parent_sha256 = _require_string(
        parent_checkpoint,
        "sha256",
        "embedding_metadata.parent_checkpoint",
    )

    if expected_parent_sha256 != two_tower_checkpoint_sha256:
        raise RuntimeError(
            "Two-Tower item embeddings were not exported from "
            "the currently loaded Two-Tower checkpoint."
        )

    metadata_lineage = _require_dict(
        metadata.get("lineage"),
        "embedding_metadata.lineage",
    )

    if metadata_lineage != expected_lineage:
        raise RuntimeError(
            "Two-Tower embedding lineage does not match the processed dataset."
        )

    item_metadata = _require_dict(
        metadata.get("item_embeddings"),
        "embedding_metadata.item_embeddings",
    )

    expected_item_embeddings_sha256 = _require_string(
        item_metadata,
        "sha256",
        "embedding_metadata.item_embeddings",
    )

    actual_item_embeddings_sha256 = sha256_file(model_release.item_embeddings_path)

    if actual_item_embeddings_sha256 != expected_item_embeddings_sha256:
        raise RuntimeError(
            "Two-Tower item embedding file fingerprint does not match its metadata."
        )

    model_config = _require_dict(
        two_tower_checkpoint.get("model_config"),
        "Two-Tower.model_config",
    )

    embedding_dim = _require_int(
        model_config,
        "embedding_dim",
        "Two-Tower.model_config",
    )

    expected_shape = [
        n_items,
        embedding_dim,
    ]

    if item_metadata.get("shape") != expected_shape:
        raise RuntimeError(
            "Two-Tower item embedding metadata shape does not "
            "match the checkpoint/catalog."
        )

    item_embeddings = load_tensor(
        path=model_release.item_embeddings_path,
        device=device,
    ).to(
        dtype=torch.float32,
    )

    if item_embeddings.ndim != 2:
        raise RuntimeError("Two-Tower item embeddings must be a 2D tensor.")

    if list(item_embeddings.shape) != expected_shape:
        raise RuntimeError(
            "Two-Tower item embedding tensor shape does not "
            "match its checkpoint/catalog."
        )

    if not torch.isfinite(item_embeddings).all():
        raise RuntimeError("Two-Tower item embeddings contain non-finite values.")

    return (
        item_embeddings,
        actual_item_embeddings_sha256,
    )


def main() -> None:
    started = time.perf_counter()

    set_seed(RANDOM_SEED)

    device = get_torch_device()

    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Load final evaluation data
    # ------------------------------------------------------------------

    train = load_parquet(TRAIN_PATH).select(
        "user_idx",
        "item_idx",
    )

    validation = load_parquet(VALIDATION_PATH).select(
        "user_idx",
        "item_idx",
    )

    test = load_parquet(TEST_PATH).select(
        "user_idx",
        "item_idx",
    )

    user_mapping = load_parquet(USER_MAPPING_PATH)

    item_mapping = load_parquet(ITEM_MAPPING_PATH)

    n_users = user_mapping.height

    n_items = item_mapping.height

    if n_users <= 0:
        raise RuntimeError("User mapping is empty.")

    if n_items <= 0:
        raise RuntimeError("Item mapping is empty.")

    (
        model_lineage,
        evaluation_lineage,
    ) = load_and_validate_evaluation_lineage(
        validation=validation,
        user_mapping=user_mapping,
        item_mapping=item_mapping,
    )

    model_release = resolve_current_model_release()

    if model_release.lineage != model_lineage:
        mismatched_keys = sorted(
            key
            for key in set(model_release.lineage) | set(model_lineage)
            if model_release.lineage.get(key) != model_lineage.get(key)
        )

        raise RuntimeError(
            "Current model release lineage does not match the "
            "processed evaluation dataset. Mismatched fields: "
            f"{mismatched_keys}"
        )

    release_catalog = _require_dict(
        model_release.manifest.get("catalog"),
        "model_release_manifest.catalog",
    )

    if (
        _require_int(
            release_catalog,
            "n_users",
            "model_release_manifest.catalog",
        )
        != n_users
    ):
        raise RuntimeError(
            "Current model release user vocabulary does not "
            "match the evaluation dataset."
        )

    if (
        _require_int(
            release_catalog,
            "n_items",
            "model_release_manifest.catalog",
        )
        != n_items
    ):
        raise RuntimeError(
            "Current model release item vocabulary does not "
            "match the evaluation dataset."
        )

    validate_test_targets(
        test=test,
        n_users=n_users,
        n_items=n_items,
    )

    (
        seen_interactions,
        warm_validation_count,
        cold_validation_count,
    ) = prepare_seen_interactions(
        train=train,
        validation=validation,
        n_users=n_users,
        n_items=n_items,
    )

    final_training_interactions = train.height + warm_validation_count

    seen_items = build_seen_items_csr(
        interactions=seen_interactions,
        n_users=n_users,
    )

    target_items = build_target_array(
        holdout=test,
        n_users=n_users,
    )

    if target_items.shape != (n_users,):
        raise RuntimeError("Target array has an invalid shape.")

    if np.any(target_items < COLD_ITEM_INDEX):
        raise RuntimeError("Test target array contains invalid item indices.")

    known_target_mask = target_items >= 0

    if np.any(target_items[known_target_mask] >= n_items):
        raise RuntimeError(
            "Test target array contains known items outside the model vocabulary."
        )

    cold_test_targets = int(np.sum(target_items == COLD_ITEM_INDEX))

    warm_test_targets = n_users - cold_test_targets

    cold_test_target_pct = cold_test_targets / n_users * 100

    print("\nEvaluation population:")

    print(f"Users                   : {n_users:,}")
    print(f"Items                   : {n_items:,}")
    print(f"Test interactions       : {test.height:,}")
    print(f"Warm test targets       : {warm_test_targets:,}")
    print(f"Cold test targets       : {cold_test_targets:,}")
    print(f"Cold test target pct    : {cold_test_target_pct:.3f}%")
    print(f"Warm validation in seen : {warm_validation_count:,}")
    print(f"Cold validation excluded: {cold_validation_count:,}")
    print(f"Data run ID             : {model_lineage['data_run_id']}")
    print(f"Release lineage ID      : {model_lineage['release_lineage_id']}")
    print(f"Model release ID        : {model_release.model_release_id}")

    # ------------------------------------------------------------------
    # Load and lineage-validate final production models
    # ------------------------------------------------------------------

    print("\nLoading final models...")

    (
        two_tower,
        two_tower_checkpoint,
        two_tower_checkpoint_sha256,
    ) = load_two_tower(
        device=device,
        model_release=model_release,
        expected_lineage=model_lineage,
        n_users=n_users,
        n_items=n_items,
        final_training_interactions=final_training_interactions,
        warm_validation_interactions=warm_validation_count,
        excluded_cold_validation_interactions=cold_validation_count,
    )

    (
        neumf,
        _neumf_checkpoint,
        neumf_checkpoint_sha256,
    ) = load_neumf(
        device=device,
        model_release=model_release,
        expected_lineage=model_lineage,
        n_users=n_users,
        n_items=n_items,
        final_training_interactions=final_training_interactions,
        warm_validation_interactions=warm_validation_count,
        excluded_cold_validation_interactions=cold_validation_count,
    )

    (
        item_embeddings,
        item_embeddings_sha256,
    ) = load_and_validate_two_tower_item_embeddings(
        device=device,
        model_release=model_release,
        two_tower_checkpoint=two_tower_checkpoint,
        two_tower_checkpoint_sha256=two_tower_checkpoint_sha256,
        expected_lineage=model_lineage,
        n_items=n_items,
    )

    # ------------------------------------------------------------------
    # Retrieval-only evaluation
    # ------------------------------------------------------------------

    print("\n=== Two-Tower Retrieval ===")

    retrieval_metrics = evaluate_embedding_retrieval(
        model=two_tower,
        item_embeddings=item_embeddings,
        target_items=target_items,
        seen_items=seen_items,
        n_users=n_users,
        batch_size=EVAL_BATCH_SIZE,
        ks=EVAL_KS,
        device=device,
        desc="Final Two-Tower test evaluation",
    )

    print(
        json.dumps(
            rounded(retrieval_metrics),
            indent=2,
        )
    )

    # ------------------------------------------------------------------
    # Retrieval + NeuMF ranking
    # ------------------------------------------------------------------

    print("\n=== Two-Tower + NeuMF ===")

    reranking_results = evaluate_retrieval_reranking(
        retrieval_model=two_tower,
        ranker=neumf,
        item_embeddings=item_embeddings,
        target_items=target_items,
        seen_items=seen_items,
        n_users=n_users,
        user_batch_size=EVAL_BATCH_SIZE,
        candidate_size=RANKING_CANDIDATE_SIZE,
        ks=EVAL_KS,
        score_batch_size=RANKING_SCORE_BATCH_SIZE,
        device=device,
        desc="Final Two-Tower + NeuMF test evaluation",
    )

    print(
        json.dumps(
            rounded(reranking_results),
            indent=2,
        )
    )

    # ------------------------------------------------------------------
    # Persist final evaluation
    # ------------------------------------------------------------------

    results = {
        "split": "test",
        "evaluation_protocol": PREPROCESSING_PROTOCOL_VERSION,
        "training_data": FINAL_TRAINING_PROTOCOL,
        "catalog_source": "training_only",
        "cold_item_index": COLD_ITEM_INDEX,
        "cold_item_policy": "retain_as_evaluation_miss",
        "n_users": n_users,
        "n_items": n_items,
        "lineage": {
            "model_lineage": model_lineage,
            "evaluation_data": evaluation_lineage,
            "model_release": {
                "model_release_id": model_release.model_release_id,
                "model_release_fingerprint": (model_release.model_release_fingerprint),
                "manifest_sha256": sha256_file(model_release.manifest_path),
            },
            "artifacts": {
                "two_tower_checkpoint_sha256": (two_tower_checkpoint_sha256),
                "neumf_checkpoint_sha256": neumf_checkpoint_sha256,
                "two_tower_item_embeddings_sha256": (item_embeddings_sha256),
            },
        },
        "evaluation_population": {
            "total_test_targets": n_users,
            "warm_test_targets": warm_test_targets,
            "cold_test_targets": cold_test_targets,
            "cold_test_target_pct": cold_test_target_pct,
        },
        "final_training_history": {
            "final_training_interactions": final_training_interactions,
            "warm_validation_interactions": warm_validation_count,
            "excluded_cold_validation_interactions": cold_validation_count,
        },
        "candidate_size": RANKING_CANDIDATE_SIZE,
        "ks": list(EVAL_KS),
        "retrieval": retrieval_metrics,
        "retrieval_reranking": reranking_results,
    }

    save_json(
        results,
        RANKING_RESULTS_PATH,
    )

    elapsed = time.perf_counter() - started

    print(f"\nSaved evaluation results:\n{RANKING_RESULTS_PATH}")

    print(f"\nTotal evaluation time: {elapsed / 60:.3f} minutes")


if __name__ == "__main__":
    main()
