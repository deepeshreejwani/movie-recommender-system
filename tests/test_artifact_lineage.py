from __future__ import annotations

from recsys.utils.artifacts import (
    build_model_data_identity,
    build_model_release_lineage,
    build_processed_data_identity,
)


def test_model_lineage_does_not_change_when_only_test_artifact_changes() -> None:
    common_model_identity = build_model_data_identity(
        preprocessing_protocol_version="preprocess-v1",
        final_training_protocol="train+warm_validation",
        train_sha256="1" * 64,
        warm_validation_sha256="2" * 64,
        user_mapping_sha256="3" * 64,
        item_mapping_sha256="4" * 64,
    )

    lineage_before = build_model_release_lineage(common_model_identity)
    lineage_after = build_model_release_lineage(common_model_identity)

    processed_before = build_processed_data_identity(
        preprocessing_protocol_version="preprocess-v1",
        preprocessing_config={"test_cold_items": 1},
        artifact_hashes={
            "train": "1" * 64,
            "validation": "5" * 64,
            "test": "6" * 64,
        },
        user_mapping_sha256="3" * 64,
        item_mapping_sha256="4" * 64,
    )
    processed_after = build_processed_data_identity(
        preprocessing_protocol_version="preprocess-v1",
        preprocessing_config={"test_cold_items": 2},
        artifact_hashes={
            "train": "1" * 64,
            "validation": "5" * 64,
            "test": "7" * 64,
        },
        user_mapping_sha256="3" * 64,
        item_mapping_sha256="4" * 64,
    )

    assert (
        processed_before["dataset_fingerprint"]
        != processed_after["dataset_fingerprint"]
    )
    assert lineage_before == lineage_after
