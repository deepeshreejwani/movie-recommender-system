from __future__ import annotations

from typing import Any

import polars as pl

from recsys.config import (
    ARTIFACT_MANIFEST_PATH,
    COLD_ITEM_INDEX,
    FINAL_TRAINING_PROTOCOL,
    ITEM_MAPPING_PATH,
    MIN_ITEM_INTERACTIONS,
    MIN_USER_INTERACTIONS,
    MOVIE_METADATA_PATH,
    MOVIES_PATH,
    POSITIVE_RATING_THRESHOLD,
    PREPROCESSING_CONFIG_PATH,
    PREPROCESSING_PROTOCOL_VERSION,
    PROCESSED_ARTIFACT_MANIFEST_VERSION,
    PROCESSED_DATA_DIR,
    RATINGS_PATH,
    TEST_PATH,
    TRAIN_PATH,
    USER_MAPPING_PATH,
    VALIDATION_PATH,
)
from recsys.data.loader import (
    load_movies,
    load_ratings,
    save_json,
    save_parquet,
)
from recsys.data.preprocessing import (
    PreparedTemporalDataset,
    prepare_temporal_dataset,
    to_implicit_feedback,
)
from recsys.utils.artifacts import (
    build_model_data_identity,
    build_processed_data_identity,
    canonical_id_mapping_sha256,
    canonical_interaction_pairs_sha256,
    file_artifact_metadata,
    sha256_file,
)

FILTER_HISTORY_PATH = PROCESSED_DATA_DIR / "filter_history.parquet"

TEMPORAL_DIAGNOSTICS_PATH = PROCESSED_DATA_DIR / "temporal_diagnostics.json"

SPLIT_VALIDATION_PATH = PROCESSED_DATA_DIR / "split_validation.json"


def build_movie_metadata(
    movies: pl.DataFrame,
    item_mapping: pl.DataFrame,
) -> pl.DataFrame:
    """
    Build movie metadata for exactly the frozen training catalog.

    The returned rows prove the relationship:

        item_idx -> movieId -> title / genres
    """

    required_movie_columns = {
        "movieId",
        "title",
        "genres",
    }

    missing_movie_columns = required_movie_columns - set(movies.columns)

    if missing_movie_columns:
        raise ValueError(
            f"Movies data is missing columns: {sorted(missing_movie_columns)}"
        )

    required_mapping_columns = {
        "movieId",
        "item_idx",
    }

    missing_mapping_columns = required_mapping_columns - set(item_mapping.columns)

    if missing_mapping_columns:
        raise ValueError(
            f"Item mapping is missing columns: {sorted(missing_mapping_columns)}"
        )

    if movies["movieId"].null_count() > 0:
        raise ValueError("Movies data contains null movieId values.")

    if movies["movieId"].n_unique() != movies.height:
        raise ValueError("Movies data contains duplicate movieId values.")

    metadata = (
        item_mapping.join(
            movies,
            on="movieId",
            how="left",
        )
        .select(
            "item_idx",
            "movieId",
            "title",
            "genres",
        )
        .sort("item_idx")
    )

    if metadata.height != item_mapping.height:
        raise RuntimeError("Movie metadata row count does not match the item mapping.")

    for column in (
        "item_idx",
        "movieId",
        "title",
        "genres",
    ):
        if metadata[column].null_count() > 0:
            raise RuntimeError(f"Movie metadata contains null values in '{column}'.")

    if metadata["item_idx"].n_unique() != item_mapping.height:
        raise RuntimeError("Movie metadata contains duplicate item_idx values.")

    if metadata["movieId"].n_unique() != item_mapping.height:
        raise RuntimeError("Movie metadata contains duplicate movieId values.")

    expected_item_indices = pl.Series(
        "item_idx",
        range(item_mapping.height),
        dtype=pl.Int32,
    )

    if not (metadata["item_idx"] == expected_item_indices).all():
        raise RuntimeError(
            "Movie metadata item_idx values are not contiguous and zero-based."
        )

    mapping_pairs = item_mapping.select(
        "item_idx",
        "movieId",
    ).sort("item_idx")

    metadata_pairs = metadata.select(
        "item_idx",
        "movieId",
    )

    if not mapping_pairs.equals(metadata_pairs):
        raise RuntimeError(
            "Movie metadata does not preserve the exact item_idx -> movieId mapping."
        )

    return metadata


def validate_prepared_dataset(
    prepared: PreparedTemporalDataset,
) -> None:
    """Validate invariants before processed artifacts are persisted."""

    n_users = prepared.user_mapping.height

    n_items = prepared.item_mapping.height

    if n_users < 1:
        raise RuntimeError("Processed user vocabulary is empty.")

    if n_items < 1:
        raise RuntimeError("Processed item vocabulary is empty.")

    if prepared.train.is_empty():
        raise RuntimeError("Processed training data is empty.")

    if prepared.validation.height != n_users:
        raise RuntimeError("Validation must contain exactly one target per user.")

    if prepared.test.height != n_users:
        raise RuntimeError("Test must contain exactly one target per user.")

    if prepared.validation["user_idx"].n_unique() != n_users:
        raise RuntimeError(
            "Validation user coverage does not match the training cohort."
        )

    if prepared.test["user_idx"].n_unique() != n_users:
        raise RuntimeError("Test user coverage does not match the training cohort.")

    if prepared.train.filter(pl.col("user_idx") < 0).height > 0:
        raise RuntimeError("Training contains negative user indices.")

    if prepared.train.filter(pl.col("item_idx") < 0).height > 0:
        raise RuntimeError("Training contains negative item indices.")

    if prepared.train.filter(pl.col("user_idx") >= n_users).height > 0:
        raise RuntimeError("Training contains users outside the frozen vocabulary.")

    if prepared.train.filter(pl.col("item_idx") >= n_items).height > 0:
        raise RuntimeError("Training contains items outside the frozen vocabulary.")

    for (
        frame_name,
        frame,
    ) in (
        (
            "validation",
            prepared.validation,
        ),
        (
            "test",
            prepared.test,
        ),
    ):
        if frame.filter(pl.col("user_idx") < 0).height > 0:
            raise RuntimeError(f"{frame_name} contains unknown users.")

        if frame.filter(pl.col("item_idx") < COLD_ITEM_INDEX).height > 0:
            raise RuntimeError(
                f"{frame_name} contains item indices below {COLD_ITEM_INDEX}."
            )

        if frame.filter(pl.col("item_idx") >= n_items).height > 0:
            raise RuntimeError(
                f"{frame_name} contains items outside the frozen vocabulary."
            )

    if not bool(
        prepared.split_validation.get(
            "valid",
            False,
        )
    ):
        raise RuntimeError("Final temporal split validation failed.")


def build_preprocessing_config(
    *,
    raw_rating_rows: int,
    positive_rows: int,
    prepared: PreparedTemporalDataset,
) -> dict[str, Any]:
    """Build deterministic preprocessing configuration metadata."""

    validation_cold_items = prepared.validation.filter(
        pl.col("item_idx") == COLD_ITEM_INDEX
    ).height

    test_cold_items = prepared.test.filter(pl.col("item_idx") == COLD_ITEM_INDEX).height

    return {
        "preprocessing_protocol_version": (PREPROCESSING_PROTOCOL_VERSION),
        "positive_rating_threshold": (POSITIVE_RATING_THRESHOLD),
        "min_user_interactions": (MIN_USER_INTERACTIONS),
        "min_item_interactions": (MIN_ITEM_INTERACTIONS),
        "cold_item_index": (COLD_ITEM_INDEX),
        "split_strategy": ("strict_per_user_temporal_leave_two_out"),
        "filtering_scope": ("provisional_train_only"),
        "mapping_scope": ("filtered_train_only"),
        "holdout_unknown_item_policy": ("retain_and_encode_as_cold_item_index"),
        "raw_rating_rows": (raw_rating_rows),
        "positive_interaction_rows": (positive_rows),
        "train_interactions": (prepared.train.height),
        "validation_interactions": (prepared.validation.height),
        "test_interactions": (prepared.test.height),
        "n_users": (prepared.user_mapping.height),
        "n_items": (prepared.item_mapping.height),
        "validation_cold_items": (validation_cold_items),
        "test_cold_items": (test_cold_items),
    }


def main() -> None:
    # ------------------------------------------------------------------
    # Load immutable MovieLens source files
    # ------------------------------------------------------------------

    print("Loading raw MovieLens data...")

    ratings = load_ratings(RATINGS_PATH)

    movies = load_movies(MOVIES_PATH)

    raw_rating_rows = ratings.height

    print(f"Raw ratings              : {raw_rating_rows:,}")

    # ------------------------------------------------------------------
    # Explicit -> positive implicit feedback
    # ------------------------------------------------------------------

    positive = to_implicit_feedback(
        interactions=ratings,
        rating_threshold=(POSITIVE_RATING_THRESHOLD),
    )

    positive_rows = positive.height

    if positive_rows < 1:
        raise RuntimeError("Positive-feedback filtering removed all interactions.")

    print(f"Positive interactions    : {positive_rows:,}")

    # Raw explicit ratings are no longer needed.
    del ratings

    # ------------------------------------------------------------------
    # Strict temporal preprocessing
    #
    # strict leave-two-out
    #       ↓
    # provisional train
    #       ↓
    # train-only iterative k-core
    #       ↓
    # training-derived cohort + mappings
    #       ↓
    # cold holdout items encoded as -1
    # ------------------------------------------------------------------

    prepared = prepare_temporal_dataset(
        interactions=positive,
        min_user_interactions=(MIN_USER_INTERACTIONS),
        min_item_interactions=(MIN_ITEM_INTERACTIONS),
        unknown_item_index=(COLD_ITEM_INDEX),
    )

    del positive

    validate_prepared_dataset(prepared)

    # ------------------------------------------------------------------
    # Build metadata for exactly the frozen item vocabulary
    # ------------------------------------------------------------------

    movie_metadata = build_movie_metadata(
        movies=movies,
        item_mapping=(prepared.item_mapping),
    )

    del movies

    preprocessing_config = build_preprocessing_config(
        raw_rating_rows=(raw_rating_rows),
        positive_rows=(positive_rows),
        prepared=prepared,
    )

    # ------------------------------------------------------------------
    # Persist canonical processed artifacts
    # ------------------------------------------------------------------

    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_parquet(
        prepared.train,
        TRAIN_PATH,
    )

    save_parquet(
        prepared.validation,
        VALIDATION_PATH,
    )

    save_parquet(
        prepared.test,
        TEST_PATH,
    )

    save_parquet(
        prepared.user_mapping,
        USER_MAPPING_PATH,
    )

    save_parquet(
        prepared.item_mapping,
        ITEM_MAPPING_PATH,
    )

    save_parquet(
        movie_metadata,
        MOVIE_METADATA_PATH,
    )

    save_parquet(
        prepared.filter_history,
        FILTER_HISTORY_PATH,
    )

    save_json(
        dict(prepared.temporal_diagnostics),
        TEMPORAL_DIAGNOSTICS_PATH,
    )

    save_json(
        dict(prepared.split_validation),
        SPLIT_VALIDATION_PATH,
    )

    save_json(
        preprocessing_config,
        PREPROCESSING_CONFIG_PATH,
    )

    # ------------------------------------------------------------------
    # Semantic mapping fingerprints
    #
    # These bind encoded indices to their original MovieLens IDs.
    # Same vocabulary SIZE is insufficient; ordering matters.
    # ------------------------------------------------------------------

    user_mapping_sha256 = canonical_id_mapping_sha256(
        mapping=(prepared.user_mapping),
        raw_id_col="userId",
        index_col="user_idx",
    )

    item_mapping_sha256 = canonical_id_mapping_sha256(
        mapping=(prepared.item_mapping),
        raw_id_col="movieId",
        index_col="item_idx",
    )

    # ------------------------------------------------------------------
    # Physical artifact fingerprints
    # ------------------------------------------------------------------

    artifact_paths = {
        "train": TRAIN_PATH,
        "validation": (VALIDATION_PATH),
        "test": TEST_PATH,
        "user_mapping": (USER_MAPPING_PATH),
        "item_mapping": (ITEM_MAPPING_PATH),
        "movie_metadata": (MOVIE_METADATA_PATH),
    }

    artifact_hashes = {
        artifact_name: (sha256_file(artifact_path))
        for (
            artifact_name,
            artifact_path,
        ) in artifact_paths.items()
    }

    # ------------------------------------------------------------------
    # Build deterministic processed-data identity
    # ------------------------------------------------------------------

    processed_identity = build_processed_data_identity(
        preprocessing_protocol_version=(PREPROCESSING_PROTOCOL_VERSION),
        preprocessing_config=(preprocessing_config),
        artifact_hashes=(artifact_hashes),
        user_mapping_sha256=(user_mapping_sha256),
        item_mapping_sha256=(item_mapping_sha256),
    )

    warm_validation = prepared.validation.filter(pl.col("item_idx") >= 0)

    warm_validation_sha256 = canonical_interaction_pairs_sha256(
        interactions=warm_validation,
    )

    model_data_identity = build_model_data_identity(
        preprocessing_protocol_version=PREPROCESSING_PROTOCOL_VERSION,
        final_training_protocol=FINAL_TRAINING_PROTOCOL,
        train_sha256=artifact_hashes["train"],
        warm_validation_sha256=warm_validation_sha256,
        user_mapping_sha256=user_mapping_sha256,
        item_mapping_sha256=item_mapping_sha256,
    )

    # ------------------------------------------------------------------
    # Full provenance manifest
    #
    # Source hashes are provenance information but are deliberately
    # separate from the processed-data semantic identity.
    # ------------------------------------------------------------------

    manifest = {
        "manifest_version": (PROCESSED_ARTIFACT_MANIFEST_VERSION),
        "data_run_id": (processed_identity["data_run_id"]),
        "dataset_fingerprint": (processed_identity["dataset_fingerprint"]),
        "preprocessing_protocol_version": (PREPROCESSING_PROTOCOL_VERSION),
        "source_artifacts": {
            "ratings": (file_artifact_metadata(RATINGS_PATH)),
            "movies": (file_artifact_metadata(MOVIES_PATH)),
        },
        "processed_artifacts": {
            artifact_name: (file_artifact_metadata(artifact_path))
            for (
                artifact_name,
                artifact_path,
            ) in artifact_paths.items()
        },
        "diagnostic_artifacts": {
            "filter_history": (file_artifact_metadata(FILTER_HISTORY_PATH)),
            "temporal_diagnostics": (file_artifact_metadata(TEMPORAL_DIAGNOSTICS_PATH)),
            "split_validation": (file_artifact_metadata(SPLIT_VALIDATION_PATH)),
            "preprocessing_config": (file_artifact_metadata(PREPROCESSING_CONFIG_PATH)),
        },
        "semantic_mapping_hashes": {
            "user_mapping_sha256": (user_mapping_sha256),
            "item_mapping_sha256": (item_mapping_sha256),
        },
        "preprocessing_config": (preprocessing_config),
        "identity": (processed_identity["identity"]),
        "model_data_identity": model_data_identity,
    }

    save_json(
        manifest,
        ARTIFACT_MANIFEST_PATH,
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    validation_cold = preprocessing_config["validation_cold_items"]

    test_cold = preprocessing_config["test_cold_items"]

    print()
    print("Processed dataset complete")
    print("--------------------------")

    print(f"Data run ID              : {processed_identity['data_run_id']}")

    print(f"Dataset fingerprint      : {processed_identity['dataset_fingerprint']}")

    print(f"Users                    : {prepared.user_mapping.height:,}")

    print(f"Items                    : {prepared.item_mapping.height:,}")

    print(f"Train interactions       : {prepared.train.height:,}")

    print(f"Validation interactions  : {prepared.validation.height:,}")

    print(f"Validation cold targets  : {validation_cold:,}")

    print(f"Test interactions        : {prepared.test.height:,}")

    print(f"Test cold targets        : {test_cold:,}")

    print(f"User mapping SHA-256     : {user_mapping_sha256}")

    print(f"Item mapping SHA-256     : {item_mapping_sha256}")

    print(f"Manifest                 : {ARTIFACT_MANIFEST_PATH}")


if __name__ == "__main__":
    main()
