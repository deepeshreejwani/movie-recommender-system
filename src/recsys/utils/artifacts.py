from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from recsys.config import PROJECT_ROOT

SHA256_HEX_LENGTH = 64

PROCESSED_IDENTITY_SCHEMA_VERSION = 1
MODEL_DATA_IDENTITY_SCHEMA_VERSION = 1


def require_metadata_dict(
    value: object,
    field_name: str,
) -> dict[str, Any]:
    """Require a dictionary in an artifact or release metadata contract."""

    if not isinstance(value, dict):
        raise TypeError(f"{field_name} must be a dictionary.")

    return value


def require_metadata_string(
    metadata: dict[str, Any],
    key: str,
    metadata_name: str,
) -> str:
    """Require a non-empty string field in metadata."""

    value = metadata.get(key)

    if not isinstance(value, str) or not value:
        raise ValueError(f"{metadata_name}['{key}'] must be a non-empty string.")

    return value


def require_metadata_int(
    metadata: dict[str, Any],
    key: str,
    metadata_name: str,
) -> int:
    """Require an integer field, preserving the existing isinstance contract."""

    value = metadata.get(key)

    if not isinstance(value, int):
        raise TypeError(f"{metadata_name}['{key}'] must be an integer.")

    return value


def require_metadata_list(
    metadata: dict[str, Any],
    key: str,
    metadata_name: str,
) -> list[Any]:
    """Require a list field in metadata."""

    value = metadata.get(key)

    if not isinstance(value, list):
        raise TypeError(f"{metadata_name}['{key}'] must be a list.")

    return value


def sha256_file(
    path: str | Path,
    chunk_size: int = 1024 * 1024,
) -> str:
    """Compute SHA-256 for a file without loading it fully into memory."""

    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1.")

    artifact_path = Path(path)

    if not artifact_path.exists():
        raise FileNotFoundError(f"Artifact does not exist: {artifact_path}")

    if not artifact_path.is_file():
        raise ValueError(f"Artifact path is not a file: {artifact_path}")

    digest = hashlib.sha256()

    with artifact_path.open("rb") as file:
        while True:
            chunk = file.read(chunk_size)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def validate_processed_manifest_file_hash(
    processed_artifacts: dict[str, Any],
    artifact_name: str,
    path: str | Path,
) -> str:
    """Validate one consumed processed file against its manifest hash.

    This deliberately preserves the existing error types and messages used
    by training, evaluation, and serving-artifact publication.
    """

    metadata_name = f"processed_artifacts[{artifact_name!r}]"
    metadata = require_metadata_dict(
        processed_artifacts.get(artifact_name),
        metadata_name,
    )
    expected_hash = require_metadata_string(
        metadata,
        "sha256",
        metadata_name,
    )

    actual_hash = sha256_file(path)

    if actual_hash != expected_hash:
        raise RuntimeError(
            f"Processed artifact fingerprint mismatch for '{artifact_name}'. "
            f"manifest={expected_hash}, actual={actual_hash}."
        )

    return actual_hash


def validate_sha256(
    value: object,
    name: str,
) -> str:
    """Validate a lowercase hexadecimal SHA-256 digest."""

    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")

    if len(value) != SHA256_HEX_LENGTH:
        raise ValueError(
            f"{name} must contain {SHA256_HEX_LENGTH} hexadecimal characters."
        )

    if value != value.lower():
        raise ValueError(f"{name} must use lowercase hexadecimal characters.")

    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} is not a valid hexadecimal SHA-256 digest.")

    return value


def canonical_json_bytes(
    data: Any,
) -> bytes:
    """
    Serialize JSON-compatible data deterministically.

    This representation is used only for identity/fingerprint generation.
    """

    serialized = json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )

    return serialized.encode("utf-8")


def canonical_json_sha256(
    data: Any,
) -> str:
    """Compute SHA-256 over deterministic JSON serialization."""

    return hashlib.sha256(canonical_json_bytes(data)).hexdigest()


def relative_project_path(
    path: str | Path,
) -> str:
    """Return a project-relative path when the artifact is inside the project."""

    resolved = Path(path).resolve()

    try:
        return str(resolved.relative_to(PROJECT_ROOT))

    except ValueError:
        return str(resolved)


def file_artifact_metadata(
    path: str | Path,
) -> dict[str, int | str]:
    """Return immutable file metadata used by artifact manifests."""

    artifact_path = Path(path)

    if not artifact_path.exists():
        raise FileNotFoundError(f"Artifact does not exist: {artifact_path}")

    if not artifact_path.is_file():
        raise ValueError(f"Artifact path is not a file: {artifact_path}")

    return {
        "path": relative_project_path(artifact_path),
        "size_bytes": artifact_path.stat().st_size,
        "sha256": sha256_file(artifact_path),
    }


def canonical_id_mapping_sha256(
    mapping: pl.DataFrame,
    raw_id_col: str,
    index_col: str,
) -> str:
    """
    Compute a semantic fingerprint for an integer ID mapping.

    The fingerprint is independent of the Parquet byte representation.

    Rows are ordered by the encoded index and both integer columns are
    normalized to signed 64-bit little-endian values before hashing.
    """

    if mapping.is_empty():
        raise ValueError("mapping cannot be empty.")

    required_columns = {
        raw_id_col,
        index_col,
    }

    missing_columns = required_columns - set(mapping.columns)

    if missing_columns:
        raise ValueError(f"Mapping is missing columns: {sorted(missing_columns)}")

    for column in required_columns:
        if mapping[column].null_count() > 0:
            raise ValueError(f"Mapping contains null values in '{column}'.")

        if not mapping[column].dtype.is_integer():
            raise TypeError(f"Mapping column '{column}' must contain integers.")

    if mapping[raw_id_col].n_unique() != mapping.height:
        raise ValueError(f"Mapping contains duplicate '{raw_id_col}' values.")

    if mapping[index_col].n_unique() != mapping.height:
        raise ValueError(f"Mapping contains duplicate '{index_col}' values.")

    ordered = mapping.select(
        raw_id_col,
        index_col,
    ).sort(index_col)

    expected_indices = np.arange(
        ordered.height,
        dtype=np.int64,
    )

    actual_indices = ordered[index_col].cast(pl.Int64).to_numpy()

    if not np.array_equal(
        actual_indices,
        expected_indices,
    ):
        raise ValueError("Mapping indices must be contiguous and zero-based.")

    raw_ids = np.ascontiguousarray(
        ordered[raw_id_col].cast(pl.Int64).to_numpy(),
        dtype="<i8",
    )

    encoded_indices = np.ascontiguousarray(
        actual_indices,
        dtype="<i8",
    )

    header = {
        "fingerprint_type": "integer_id_mapping",
        "fingerprint_version": 1,
        "raw_id_col": raw_id_col,
        "index_col": index_col,
        "rows": ordered.height,
    }

    digest = hashlib.sha256()

    digest.update(canonical_json_bytes(header))

    digest.update(b"\x00raw_ids\x00")
    digest.update(raw_ids.tobytes(order="C"))

    digest.update(b"\x00encoded_indices\x00")
    digest.update(encoded_indices.tobytes(order="C"))

    return digest.hexdigest()


def canonical_interaction_pairs_sha256(
    interactions: pl.DataFrame,
    user_col: str = "user_idx",
    item_col: str = "item_idx",
) -> str:
    """Hash the semantic multiset of encoded user-item interactions."""

    required_columns = {user_col, item_col}
    missing_columns = required_columns - set(interactions.columns)

    if missing_columns:
        raise ValueError(
            f"Interactions are missing required columns: {sorted(missing_columns)}"
        )

    for column in (user_col, item_col):
        if interactions[column].null_count() > 0:
            raise ValueError(f"Interactions contain null values in '{column}'.")

    ordered = interactions.select(
        pl.col(user_col).cast(pl.Int64),
        pl.col(item_col).cast(pl.Int64),
    ).sort(user_col, item_col)

    users = ordered[user_col].to_numpy().astype(np.int64, copy=False)
    items = ordered[item_col].to_numpy().astype(np.int64, copy=False)

    header = {
        "schema": "encoded_interaction_pairs_v1",
        "user_col": user_col,
        "item_col": item_col,
        "rows": ordered.height,
    }

    digest = hashlib.sha256()
    digest.update(canonical_json_bytes(header))
    digest.update(b"\x00users\x00")
    digest.update(users.tobytes(order="C"))
    digest.update(b"\x00items\x00")
    digest.update(items.tobytes(order="C"))

    return digest.hexdigest()


def build_model_data_identity(
    *,
    preprocessing_protocol_version: str,
    final_training_protocol: str,
    train_sha256: str,
    warm_validation_sha256: str,
    user_mapping_sha256: str,
    item_mapping_sha256: str,
) -> dict[str, Any]:
    """Build test-independent identity for the data that can affect model training."""

    if not preprocessing_protocol_version:
        raise ValueError("preprocessing_protocol_version cannot be empty.")

    if not final_training_protocol:
        raise ValueError("final_training_protocol cannot be empty.")

    payload = {
        "identity_schema_version": MODEL_DATA_IDENTITY_SCHEMA_VERSION,
        "preprocessing_protocol_version": preprocessing_protocol_version,
        "final_training_protocol": final_training_protocol,
        "train_sha256": validate_sha256(train_sha256, "train_sha256"),
        "warm_validation_sha256": validate_sha256(
            warm_validation_sha256,
            "warm_validation_sha256",
        ),
        "user_mapping_sha256": validate_sha256(
            user_mapping_sha256,
            "user_mapping_sha256",
        ),
        "item_mapping_sha256": validate_sha256(
            item_mapping_sha256,
            "item_mapping_sha256",
        ),
    }

    dataset_fingerprint = canonical_json_sha256(payload)

    return {
        "data_run_id": f"model-data-{dataset_fingerprint[:20]}",
        "dataset_fingerprint": dataset_fingerprint,
        "identity": payload,
    }


def validate_model_data_identity(identity: object) -> dict[str, Any]:
    """Validate test-independent model-data identity metadata."""

    if not isinstance(identity, dict):
        raise TypeError("Model data identity must be a dictionary.")

    data_run_id = identity.get("data_run_id")
    dataset_fingerprint = identity.get("dataset_fingerprint")
    payload = identity.get("identity")

    if not isinstance(data_run_id, str):
        raise TypeError("model data_run_id must be a string.")

    if not data_run_id.startswith("model-data-"):
        raise ValueError("model data_run_id has an invalid prefix.")

    dataset_fingerprint = validate_sha256(
        dataset_fingerprint,
        "model dataset_fingerprint",
    )

    if not isinstance(payload, dict):
        raise TypeError("Model data identity payload must be a dictionary.")

    if payload.get("identity_schema_version") != MODEL_DATA_IDENTITY_SCHEMA_VERSION:
        raise ValueError("Unsupported model data identity schema version.")

    for key in (
        "train_sha256",
        "warm_validation_sha256",
        "user_mapping_sha256",
        "item_mapping_sha256",
    ):
        validate_sha256(payload.get(key), f"model_data_identity.identity.{key}")

    for key in ("preprocessing_protocol_version", "final_training_protocol"):
        value = payload.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(
                f"model_data_identity.identity['{key}'] must be a non-empty string."
            )

    expected_fingerprint = canonical_json_sha256(payload)

    if dataset_fingerprint != expected_fingerprint:
        raise ValueError(
            "Model dataset fingerprint does not match its identity payload."
        )

    expected_run_id = f"model-data-{dataset_fingerprint[:20]}"

    if data_run_id != expected_run_id:
        raise ValueError("model data_run_id does not match the dataset fingerprint.")

    return identity


def build_model_release_lineage(
    model_data_identity: object,
) -> dict[str, str]:
    """Build model-release lineage solely from test-independent model data."""

    identity = validate_model_data_identity(model_data_identity)
    payload = identity["identity"]

    release_payload = {
        "data_run_id": str(identity["data_run_id"]),
        "dataset_fingerprint": str(identity["dataset_fingerprint"]),
        "preprocessing_protocol_version": str(
            payload["preprocessing_protocol_version"]
        ),
        "final_training_protocol": str(payload["final_training_protocol"]),
        "train_sha256": str(payload["train_sha256"]),
        "warm_validation_sha256": str(payload["warm_validation_sha256"]),
        "user_mapping_sha256": str(payload["user_mapping_sha256"]),
        "item_mapping_sha256": str(payload["item_mapping_sha256"]),
    }

    release_fingerprint = canonical_json_sha256(release_payload)

    return {
        **release_payload,
        "release_lineage_id": f"release-{release_fingerprint[:20]}",
        "release_fingerprint": release_fingerprint,
    }


def build_processed_data_identity(
    *,
    preprocessing_protocol_version: str,
    preprocessing_config: dict[str, Any],
    artifact_hashes: dict[str, str],
    user_mapping_sha256: str,
    item_mapping_sha256: str,
) -> dict[str, Any]:
    """
    Build the deterministic identity of one processed dataset.

    `data_run_id` is content-derived. Reproducing exactly the same
    processed dataset with the same protocol yields the same identity.
    """

    if not preprocessing_protocol_version:
        raise ValueError("preprocessing_protocol_version cannot be empty.")

    if not isinstance(
        preprocessing_config,
        dict,
    ):
        raise TypeError("preprocessing_config must be a dictionary.")

    if not artifact_hashes:
        raise ValueError("artifact_hashes cannot be empty.")

    normalized_artifact_hashes: dict[str, str] = {}

    for (
        artifact_name,
        artifact_hash,
    ) in sorted(artifact_hashes.items()):
        if not artifact_name:
            raise ValueError("Artifact names cannot be empty.")

        normalized_artifact_hashes[artifact_name] = validate_sha256(
            artifact_hash,
            f"artifact_hashes[{artifact_name!r}]",
        )

    user_mapping_sha256 = validate_sha256(
        user_mapping_sha256,
        "user_mapping_sha256",
    )

    item_mapping_sha256 = validate_sha256(
        item_mapping_sha256,
        "item_mapping_sha256",
    )

    identity_payload = {
        "identity_schema_version": (PROCESSED_IDENTITY_SCHEMA_VERSION),
        "preprocessing_protocol_version": (preprocessing_protocol_version),
        "preprocessing_config": (preprocessing_config),
        "artifact_hashes": (normalized_artifact_hashes),
        "semantic_mapping_hashes": {
            "user_mapping_sha256": (user_mapping_sha256),
            "item_mapping_sha256": (item_mapping_sha256),
        },
    }

    dataset_fingerprint = canonical_json_sha256(identity_payload)

    return {
        "data_run_id": (f"processed-{dataset_fingerprint[:20]}"),
        "dataset_fingerprint": (dataset_fingerprint),
        "identity": identity_payload,
    }


def validate_processed_data_identity(
    identity: object,
) -> dict[str, Any]:
    """Validate the minimum structure of processed lineage metadata."""

    if not isinstance(
        identity,
        dict,
    ):
        raise TypeError("Processed data identity must be a dictionary.")

    data_run_id = identity.get("data_run_id")

    dataset_fingerprint = identity.get("dataset_fingerprint")

    payload = identity.get("identity")

    if not isinstance(
        data_run_id,
        str,
    ):
        raise TypeError("data_run_id must be a string.")

    if not data_run_id.startswith("processed-"):
        raise ValueError("data_run_id has an invalid prefix.")

    dataset_fingerprint = validate_sha256(
        dataset_fingerprint,
        "dataset_fingerprint",
    )

    if not isinstance(
        payload,
        dict,
    ):
        raise TypeError("identity payload must be a dictionary.")

    if payload.get("identity_schema_version") != PROCESSED_IDENTITY_SCHEMA_VERSION:
        raise ValueError("Unsupported processed data identity schema version.")

    expected_fingerprint = canonical_json_sha256(payload)

    if dataset_fingerprint != expected_fingerprint:
        raise ValueError(
            "Processed dataset fingerprint does not match its identity payload."
        )

    expected_run_id = f"processed-{dataset_fingerprint[:20]}"

    if data_run_id != expected_run_id:
        raise ValueError("data_run_id does not match the dataset fingerprint.")

    return identity
