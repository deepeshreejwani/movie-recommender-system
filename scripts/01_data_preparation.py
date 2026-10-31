"""MovieLens 32M exploration and canonical artifact validation.

Read-only workflow:
- Inspect raw MovieLens ratings and movie metadata.
- Calculate descriptive and positive-feedback statistics.
- Verify canonical processed artifacts and their SHA-256 hashes.
- Validate mappings, split conventions, and processed-data lineage.

Does not run preprocessing, train models, or write artifacts.
"""

from __future__ import annotations

import argparse

import numpy as np
import polars as pl

from recsys.config import (
    ARTIFACT_MANIFEST_PATH,
    COLD_ITEM_INDEX,
    ITEM_MAPPING_PATH,
    MIN_ITEM_INTERACTIONS,
    MIN_USER_INTERACTIONS,
    MOVIE_METADATA_PATH,
    MOVIES_PATH,
    POSITIVE_RATING_THRESHOLD,
    PREPROCESSING_CONFIG_PATH,
    PROCESSED_DATA_DIR,
    RATINGS_PATH,
    TEST_PATH,
    TRAIN_PATH,
    USER_MAPPING_PATH,
    VALIDATION_PATH,
)
from recsys.data.loader import (
    load_json,
    load_movies,
    load_parquet,
    load_ratings,
)
from recsys.data.preprocessing import to_implicit_feedback
from recsys.utils.artifacts import (
    canonical_id_mapping_sha256,
    sha256_file,
    validate_processed_data_identity,
)

FILTER_HISTORY_PATH = PROCESSED_DATA_DIR / "filter_history.parquet"
TEMPORAL_DIAGNOSTICS_PATH = PROCESSED_DATA_DIR / "temporal_diagnostics.json"
SPLIT_VALIDATION_PATH = PROCESSED_DATA_DIR / "split_validation.json"


def display(title: str, value: object) -> None:
    """Print a labeled inspection result."""
    print(f"\n{'=' * 18} {title} {'=' * 18}")
    print(value)


def activity_summary(
    activity: pl.DataFrame,
    count_column: str,
) -> pl.DataFrame:
    """Calculate distribution statistics for user or movie activity."""
    count = pl.col(count_column)

    return activity.select(
        count.min().alias("min"),
        count.quantile(0.25).alias("p25"),
        count.median().alias("median"),
        count.mean().alias("mean"),
        count.quantile(0.75).alias("p75"),
        count.quantile(0.90).alias("p90"),
        count.quantile(0.99).alias("p99"),
        count.max().alias("max"),
    )


def inspect_raw_data() -> None:
    """Inspect raw ratings, movie metadata, and interaction distributions."""
    artifact_paths = {
        "raw_ratings": RATINGS_PATH,
        "raw_movies": MOVIES_PATH,
        "train": TRAIN_PATH,
        "validation": VALIDATION_PATH,
        "test": TEST_PATH,
        "user_mapping": USER_MAPPING_PATH,
        "item_mapping": ITEM_MAPPING_PATH,
        "movie_metadata": MOVIE_METADATA_PATH,
        "preprocessing_config": PREPROCESSING_CONFIG_PATH,
        "artifact_manifest": ARTIFACT_MANIFEST_PATH,
    }

    display(
        "Artifact inventory",
        pl.DataFrame(
            [
                {
                    "artifact": name,
                    "exists": path.exists(),
                    "path": str(path),
                }
                for name, path in artifact_paths.items()
            ]
        ),
    )

    ratings = load_ratings(RATINGS_PATH)
    movies = load_movies(MOVIES_PATH)

    print("\nRatings")
    print(f"Rows    : {ratings.height:,}")
    print(f"Columns : {ratings.width}")
    print(f"Schema  : {ratings.schema}")

    print("\nMovies")
    print(f"Rows    : {movies.height:,}")
    print(f"Columns : {movies.width}")
    print(f"Schema  : {movies.schema}")

    display("Ratings sample", ratings.head())
    display("Movies sample", movies.head())

    display("Ratings null counts", ratings.null_count())
    display("Movies null counts", movies.null_count())

    repeated_user_movie_pairs = (
        ratings.group_by(["userId", "movieId"]).len().filter(pl.col("len") > 1)
    )

    print(f"Repeated user-movie pairs: {repeated_user_movie_pairs.height:,}")

    missing_movie_metadata = (
        ratings.select("movieId")
        .unique()
        .join(
            movies.select("movieId"),
            on="movieId",
            how="anti",
        )
    )

    print(f"Rated movie IDs missing from metadata: {missing_movie_metadata.height:,}")

    summary = ratings.select(
        pl.len().alias("interactions"),
        pl.col("userId").n_unique().alias("users"),
        pl.col("movieId").n_unique().alias("rated_movies"),
        pl.col("rating").min().alias("min_rating"),
        pl.col("rating").max().alias("max_rating"),
        pl.col("rating").mean().alias("mean_rating"),
        pl.col("timestamp").min().alias("min_timestamp"),
        pl.col("timestamp").max().alias("max_timestamp"),
    )

    display("Raw dataset summary", summary)

    n_interactions = ratings.height
    n_users = ratings["userId"].n_unique()
    n_rated_movies = ratings["movieId"].n_unique()

    density = n_interactions / (n_users * n_rated_movies)
    sparsity = 1 - density

    print(f"Interactions: {n_interactions:,}")
    print(f"Users       : {n_users:,}")
    print(f"Rated movies: {n_rated_movies:,}")
    print(f"Density     : {density:.3%}")
    print(f"Sparsity    : {sparsity:.3%}")

    ratings = ratings.with_columns(
        pl.from_epoch(
            pl.col("timestamp"),
            time_unit="s",
        ).alias("event_time")
    )

    display(
        "Temporal coverage (UTC)",
        ratings.select(
            pl.col("event_time").min().alias("first_interaction"),
            pl.col("event_time").max().alias("last_interaction"),
        ),
    )

    rating_distribution = (
        ratings.group_by("rating")
        .agg(pl.len().alias("interactions"))
        .sort("rating")
        .with_columns(
            (pl.col("interactions") / pl.col("interactions").sum()).alias(
                "interaction_share"
            )
        )
        .with_columns(
            (pl.col("interaction_share") * 100).round(2).alias("interaction_share_pct")
        )
    )

    display("Rating distribution", rating_distribution)

    user_activity = ratings.group_by("userId").agg(pl.len().alias("interaction_count"))

    item_activity = ratings.group_by("movieId").agg(pl.len().alias("interaction_count"))

    display(
        "User activity",
        activity_summary(user_activity, "interaction_count"),
    )

    display(
        "Item activity",
        activity_summary(item_activity, "interaction_count"),
    )

    positive_interactions = to_implicit_feedback(
        interactions=ratings,
        rating_threshold=POSITIVE_RATING_THRESHOLD,
    )

    print(f"\nRaw interactions      : {ratings.height:,}")
    print(f"Positive interactions : {positive_interactions.height:,}")
    print(
        "Positive share        : "
        f"{positive_interactions.height / ratings.height * 100:.3f}%"
    )

    display(
        "Positive-feedback population",
        positive_interactions.select(
            pl.len().alias("interactions"),
            pl.col("userId").n_unique().alias("users"),
            pl.col("movieId").n_unique().alias("movies"),
        ),
    )

    positive_user_activity = positive_interactions.group_by("userId").agg(
        pl.len().alias("positive_interactions")
    )

    positive_item_activity = positive_interactions.group_by("movieId").agg(
        pl.len().alias("positive_interactions")
    )

    display(
        "Positive user activity",
        activity_summary(
            positive_user_activity,
            "positive_interactions",
        ),
    )

    display(
        "Positive item activity",
        activity_summary(
            positive_item_activity,
            "positive_interactions",
        ),
    )

    positive_item_popularity = positive_item_activity.sort(
        "positive_interactions",
        descending=True,
    ).with_row_index("popularity_rank", offset=1)

    total_positive_interactions = positive_item_popularity[
        "positive_interactions"
    ].sum()

    long_tail_rows = []

    for fraction in (0.01, 0.05, 0.10, 0.20):
        top_n = max(
            1,
            int(np.ceil(positive_item_popularity.height * fraction)),
        )

        top_interactions = positive_item_popularity.head(top_n)[
            "positive_interactions"
        ].sum()

        long_tail_rows.append(
            {
                "top_catalog_pct": fraction * 100,
                "movies": top_n,
                "interaction_share_pct": (
                    top_interactions / total_positive_interactions * 100
                ),
            }
        )

    long_tail_summary = pl.DataFrame(long_tail_rows).with_columns(
        pl.col("top_catalog_pct").round(3),
        pl.col("interaction_share_pct").round(3),
    )

    display(
        "Positive interaction long tail",
        long_tail_summary,
    )


def inspect_processed_data() -> None:
    """Validate existing canonical processed artifacts."""
    manifest = load_json(ARTIFACT_MANIFEST_PATH)

    diagnostic_paths = {
        "filter_history": FILTER_HISTORY_PATH,
        "temporal_diagnostics": TEMPORAL_DIAGNOSTICS_PATH,
        "split_validation": SPLIT_VALIDATION_PATH,
        "preprocessing_config": PREPROCESSING_CONFIG_PATH,
    }

    diagnostic_artifacts = manifest.get("diagnostic_artifacts")

    if not isinstance(diagnostic_artifacts, dict):
        raise TypeError("artifact_manifest.diagnostic_artifacts must be a dictionary.")

    if set(diagnostic_artifacts) != set(diagnostic_paths):
        raise RuntimeError(
            "Diagnostic manifest entries do not match the expected diagnostic files."
        )

    diagnostic_integrity_rows = []

    for name, path in diagnostic_paths.items():
        metadata = diagnostic_artifacts.get(name)

        if not isinstance(metadata, dict) or not isinstance(
            metadata.get("sha256"), str
        ):
            raise TypeError(f"Missing diagnostic artifact SHA-256 for {name}.")

        actual_sha256 = sha256_file(path)

        if actual_sha256 != metadata["sha256"]:
            raise RuntimeError(f"Diagnostic artifact fingerprint mismatch: {name}.")

        diagnostic_integrity_rows.append(
            {
                "artifact": name,
                "valid": True,
                "size_mb": path.stat().st_size / (1024**2),
            }
        )

    display(
        "Diagnostic artifact hashes",
        pl.DataFrame(diagnostic_integrity_rows).with_columns(
            pl.col("size_mb").round(3)
        ),
    )

    # Verify diagnostic hashes before loading their contents.
    preprocessing_config = load_json(PREPROCESSING_CONFIG_PATH)
    filter_history = load_parquet(FILTER_HISTORY_PATH)
    temporal_diagnostics = load_json(TEMPORAL_DIAGNOSTICS_PATH)
    split_validation = load_json(SPLIT_VALIDATION_PATH)

    train = load_parquet(TRAIN_PATH)
    validation = load_parquet(VALIDATION_PATH)
    test = load_parquet(TEST_PATH)

    user_mapping = load_parquet(USER_MAPPING_PATH)
    item_mapping = load_parquet(ITEM_MAPPING_PATH)
    model_movies = load_parquet(MOVIE_METADATA_PATH)

    canonical_summary = pl.DataFrame(
        [
            {
                "metric": "Raw ratings",
                "value": preprocessing_config["raw_rating_rows"],
            },
            {
                "metric": "Positive interactions",
                "value": preprocessing_config["positive_interaction_rows"],
            },
            {
                "metric": "Users",
                "value": preprocessing_config["n_users"],
            },
            {
                "metric": "Items",
                "value": preprocessing_config["n_items"],
            },
            {
                "metric": "Train interactions",
                "value": preprocessing_config["train_interactions"],
            },
            {
                "metric": "Validation interactions",
                "value": preprocessing_config["validation_interactions"],
            },
            {
                "metric": "Validation cold targets",
                "value": preprocessing_config["validation_cold_items"],
            },
            {
                "metric": "Test interactions",
                "value": preprocessing_config["test_interactions"],
            },
            {
                "metric": "Test cold targets",
                "value": preprocessing_config["test_cold_items"],
            },
        ]
    )

    display(
        "Canonical processed population",
        canonical_summary,
    )
    display(
        "Temporal diagnostics",
        temporal_diagnostics,
    )
    display(
        "Filter history",
        filter_history,
    )

    split_summary = pl.DataFrame(
        [
            {
                "split": "train",
                "interactions": train.height,
                "users": train["user_idx"].n_unique(),
                "warm_targets": None,
                "cold_targets": None,
                "cold_pct": None,
            },
            {
                "split": "validation",
                "interactions": validation.height,
                "users": validation["user_idx"].n_unique(),
                "warm_targets": validation.filter(pl.col("item_idx") >= 0).height,
                "cold_targets": validation.filter(
                    pl.col("item_idx") == COLD_ITEM_INDEX
                ).height,
                "cold_pct": (
                    validation.filter(pl.col("item_idx") == COLD_ITEM_INDEX).height
                    / validation.height
                    * 100
                ),
            },
            {
                "split": "test",
                "interactions": test.height,
                "users": test["user_idx"].n_unique(),
                "warm_targets": test.filter(pl.col("item_idx") >= 0).height,
                "cold_targets": test.filter(
                    pl.col("item_idx") == COLD_ITEM_INDEX
                ).height,
                "cold_pct": (
                    test.filter(pl.col("item_idx") == COLD_ITEM_INDEX).height
                    / test.height
                    * 100
                ),
            },
        ]
    ).with_columns(pl.col("cold_pct").round(3))

    display(
        "Train/validation/test summary",
        split_summary,
    )
    display(
        "Split validation",
        split_validation,
    )

    n_users = user_mapping.height
    n_items = item_mapping.height

    assert user_mapping["user_idx"].min() == 0
    assert user_mapping["user_idx"].max() == n_users - 1
    assert user_mapping["user_idx"].n_unique() == n_users

    assert item_mapping["item_idx"].min() == 0
    assert item_mapping["item_idx"].max() == n_items - 1
    assert item_mapping["item_idx"].n_unique() == n_items

    assert model_movies.height == n_items
    assert model_movies["item_idx"].n_unique() == n_items
    assert model_movies["movieId"].n_unique() == n_items

    assert train.filter(pl.col("item_idx") < 0).is_empty()

    assert validation.filter(pl.col("item_idx") < COLD_ITEM_INDEX).is_empty()

    assert test.filter(pl.col("item_idx") < COLD_ITEM_INDEX).is_empty()

    print(f"\nEncoded mappings validated: {n_users:,} users, {n_items:,} items")

    user_mapping_sha256 = canonical_id_mapping_sha256(
        mapping=user_mapping,
        raw_id_col="userId",
        index_col="user_idx",
    )

    item_mapping_sha256 = canonical_id_mapping_sha256(
        mapping=item_mapping,
        raw_id_col="movieId",
        index_col="item_idx",
    )

    expected_hashes = manifest["semantic_mapping_hashes"]

    assert user_mapping_sha256 == expected_hashes["user_mapping_sha256"]

    assert item_mapping_sha256 == expected_hashes["item_mapping_sha256"]

    processed_identity = {
        "data_run_id": manifest["data_run_id"],
        "dataset_fingerprint": manifest["dataset_fingerprint"],
        "identity": manifest["identity"],
    }

    validate_processed_data_identity(processed_identity)

    assert manifest["preprocessing_config"] == preprocessing_config

    print(f"Data run ID: {manifest['data_run_id']}")
    print(f"Dataset fingerprint: {manifest['dataset_fingerprint']}")
    print(f"User mapping SHA-256: {user_mapping_sha256}")
    print(f"Item mapping SHA-256: {item_mapping_sha256}")

    processed_paths = {
        "train": TRAIN_PATH,
        "validation": VALIDATION_PATH,
        "test": TEST_PATH,
        "user_mapping": USER_MAPPING_PATH,
        "item_mapping": ITEM_MAPPING_PATH,
        "movie_metadata": MOVIE_METADATA_PATH,
    }

    artifact_integrity = []

    for name, path in processed_paths.items():
        expected_sha256 = manifest["processed_artifacts"][name]["sha256"]

        actual_sha256 = sha256_file(path)

        artifact_integrity.append(
            {
                "artifact": name,
                "valid": (actual_sha256 == expected_sha256),
                "size_mb": (path.stat().st_size / (1024**2)),
            }
        )

    artifact_integrity_table = pl.DataFrame(artifact_integrity).with_columns(
        pl.col("size_mb").round(3)
    )

    display(
        "Processed artifact hashes",
        artifact_integrity_table,
    )

    assert artifact_integrity_table["valid"].all()

    print("\nCanonical processed data and lineage validated.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only MovieLens data exploration and processed-artifact inspection."
        )
    )

    parser.add_argument(
        "--processed-only",
        action="store_true",
        help=("Validate canonical processed files without loading raw 32M ratings."),
    )

    args = parser.parse_args()

    pl.Config.set_tbl_rows(10)
    pl.Config.set_tbl_cols(20)
    pl.Config.set_fmt_str_lengths(80)

    print(
        "Canonical preprocessing is owned by scripts/preprocess.py (not executed here)."
    )
    print(f"Positive rating threshold: {POSITIVE_RATING_THRESHOLD}")
    print(
        "Train-only k-core: "
        f"users >= {MIN_USER_INTERACTIONS}, "
        f"items >= {MIN_ITEM_INTERACTIONS}"
    )

    if not args.processed_only:
        inspect_raw_data()

    inspect_processed_data()


if __name__ == "__main__":
    main()
