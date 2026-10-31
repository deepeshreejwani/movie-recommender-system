from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from recsys.data.splits import (
    temporal_user_split,
    validate_temporal_split,
)


@dataclass
class PreparedTemporalDataset:
    train: pl.DataFrame
    validation: pl.DataFrame
    test: pl.DataFrame

    user_mapping: pl.DataFrame
    item_mapping: pl.DataFrame

    filter_history: pl.DataFrame

    temporal_diagnostics: dict[
        str,
        int | float,
    ]

    split_validation: dict[
        str,
        int | bool,
    ]

    validation_cold_items: int


def to_implicit_feedback(
    interactions: pl.DataFrame,
    rating_threshold: float = 4.0,
    rating_col: str = "rating",
) -> pl.DataFrame:
    """
    Retain interactions whose explicit rating meets
    the positive-feedback threshold.
    """

    if rating_col not in interactions.columns:
        raise ValueError(f"Missing rating column: {rating_col}")

    return interactions.filter(pl.col(rating_col) >= rating_threshold)


def iterative_interaction_filter(
    interactions: pl.DataFrame,
    min_user_interactions: int,
    min_item_interactions: int,
    user_col: str = "userId",
    item_col: str = "movieId",
) -> tuple[
    pl.DataFrame,
    pl.DataFrame,
]:
    """
    Iteratively remove low-activity users and items until every
    remaining entity satisfies the configured interaction
    requirements.
    """

    if min_user_interactions < 1:
        raise ValueError("min_user_interactions must be >= 1.")

    if min_item_interactions < 1:
        raise ValueError("min_item_interactions must be >= 1.")

    if interactions.is_empty():
        raise ValueError("interactions cannot be empty.")

    required_columns = {
        user_col,
        item_col,
    }

    missing_columns = required_columns - set(interactions.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    filtered = interactions.clone()

    history: list[dict[str, int]] = []

    iteration = 0

    while True:
        iteration += 1

        rows_before = filtered.height

        eligible_users = (
            filtered.group_by(user_col)
            .len()
            .filter(pl.col("len") >= min_user_interactions)
            .select(user_col)
        )

        filtered = filtered.join(
            eligible_users,
            on=user_col,
            how="semi",
        )

        if filtered.is_empty():
            raise RuntimeError(
                "Interaction filtering removed "
                "all interactions after applying "
                "the user threshold."
            )

        eligible_items = (
            filtered.group_by(item_col)
            .len()
            .filter(pl.col("len") >= min_item_interactions)
            .select(item_col)
        )

        filtered = filtered.join(
            eligible_items,
            on=item_col,
            how="semi",
        )

        if filtered.is_empty():
            raise RuntimeError(
                "Interaction filtering removed "
                "all interactions after applying "
                "the item threshold."
            )

        rows_after = filtered.height

        history.append(
            {
                "iteration": iteration,
                "rows_before": rows_before,
                "rows_after": rows_after,
                "removed": (rows_before - rows_after),
                "users": (filtered[user_col].n_unique()),
                "movies": (filtered[item_col].n_unique()),
            }
        )

        if rows_after == rows_before:
            break

    return (
        filtered,
        pl.DataFrame(history),
    )


def build_id_mappings(
    interactions: pl.DataFrame,
    user_col: str = "userId",
    item_col: str = "movieId",
) -> tuple[
    pl.DataFrame,
    pl.DataFrame,
]:
    """
    Build deterministic zero-based user and item mappings.
    """

    if interactions.is_empty():
        raise ValueError("interactions cannot be empty.")

    required_columns = {
        user_col,
        item_col,
    }

    missing_columns = required_columns - set(interactions.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    user_mapping = (
        interactions.select(user_col)
        .unique()
        .sort(user_col)
        .with_row_index("user_idx")
        .select(
            user_col,
            pl.col("user_idx").cast(pl.Int32),
        )
    )

    item_mapping = (
        interactions.select(item_col)
        .unique()
        .sort(item_col)
        .with_row_index("item_idx")
        .select(
            item_col,
            pl.col("item_idx").cast(pl.Int32),
        )
    )

    return (
        user_mapping,
        item_mapping,
    )


def apply_id_mappings(
    interactions: pl.DataFrame,
    user_mapping: pl.DataFrame,
    item_mapping: pl.DataFrame,
    user_col: str = "userId",
    item_col: str = "movieId",
    allow_unknown_users: bool = False,
    allow_unknown_items: bool = False,
    unknown_index: int = -1,
) -> pl.DataFrame:
    """
    Attach zero-based user and item indices.

    Unknown IDs are rejected by default.

    When explicitly allowed, unknown IDs are encoded with
    `unknown_index` instead of dropping interactions.
    """

    if unknown_index >= 0:
        raise ValueError("unknown_index must be negative.")

    required_columns = {
        user_col,
        item_col,
    }

    missing_columns = required_columns - set(interactions.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required interaction columns: {sorted(missing_columns)}"
        )

    encoded = interactions.join(
        user_mapping,
        on=user_col,
        how="left",
    ).join(
        item_mapping,
        on=item_col,
        how="left",
    )

    if encoded.height != interactions.height:
        raise RuntimeError("ID mapping changed interaction row count.")

    missing_users = encoded["user_idx"].null_count()

    missing_items = encoded["item_idx"].null_count()

    if missing_users > 0 and not allow_unknown_users:
        raise ValueError(f"Unmapped users detected: {missing_users}")

    if missing_items > 0 and not allow_unknown_items:
        raise ValueError(f"Unmapped items detected: {missing_items}")

    if allow_unknown_users:
        encoded = encoded.with_columns(
            pl.col("user_idx").fill_null(unknown_index).cast(pl.Int32)
        )

    if allow_unknown_items:
        encoded = encoded.with_columns(
            pl.col("item_idx").fill_null(unknown_index).cast(pl.Int32)
        )

    return encoded


def prepare_temporal_dataset(
    interactions: pl.DataFrame,
    min_user_interactions: int,
    min_item_interactions: int,
    user_col: str = "userId",
    item_col: str = "movieId",
    timestamp_col: str = "timestamp",
    unknown_item_index: int = -1,
) -> PreparedTemporalDataset:
    """
    Build a leakage-safe temporal recommendation dataset.

    Processing order:

    1. Strict temporal leave-two-out split.
    2. Train-only iterative interaction filtering.
    3. Retain evaluation users based only on training eligibility.
    4. Build user/item mappings from training only.
    5. Encode unseen holdout items as -1.

    Validation/test item identities never influence the
    training vocabulary or training eligibility.
    """

    if interactions.is_empty():
        raise ValueError("interactions cannot be empty.")

    if min_user_interactions < 1:
        raise ValueError("min_user_interactions must be >= 1.")

    if min_item_interactions < 1:
        raise ValueError("min_item_interactions must be >= 1.")

    if unknown_item_index != -1:
        raise ValueError(
            "prepare_temporal_dataset currently requires unknown_item_index=-1."
        )

    required_columns = {
        user_col,
        item_col,
        timestamp_col,
    }

    missing_columns = required_columns - set(interactions.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    (
        provisional_train,
        provisional_validation,
        provisional_test,
        temporal_diagnostics,
    ) = temporal_user_split(
        interactions=interactions,
        user_col=user_col,
        item_col=item_col,
        timestamp_col=timestamp_col,
        return_diagnostics=True,
    )

    (
        filtered_train,
        filter_history,
    ) = iterative_interaction_filter(
        interactions=(provisional_train),
        min_user_interactions=(min_user_interactions),
        min_item_interactions=(min_item_interactions),
        user_col=user_col,
        item_col=item_col,
    )

    if filtered_train.is_empty():
        raise RuntimeError("Training data is empty after filtering.")

    user_activity = filtered_train.group_by(user_col).len()

    item_activity = filtered_train.group_by(item_col).len()

    min_final_user_activity = user_activity["len"].min()

    min_final_item_activity = item_activity["len"].min()

    if (
        min_final_user_activity is None
        or min_final_user_activity < min_user_interactions
    ):
        raise RuntimeError(
            "Final training data violates the minimum user interaction threshold."
        )

    if (
        min_final_item_activity is None
        or min_final_item_activity < min_item_interactions
    ):
        raise RuntimeError(
            "Final training data violates the minimum item interaction threshold."
        )

    eligible_users = filtered_train.select(user_col).unique()

    validation = provisional_validation.join(
        eligible_users,
        on=user_col,
        how="semi",
    )

    test = provisional_test.join(
        eligible_users,
        on=user_col,
        how="semi",
    )

    split_validation = validate_temporal_split(
        train=filtered_train,
        validation=validation,
        test=test,
        user_col=user_col,
        item_col=item_col,
        timestamp_col=(timestamp_col),
    )

    (
        user_mapping,
        item_mapping,
    ) = build_id_mappings(
        interactions=filtered_train,
        user_col=user_col,
        item_col=item_col,
    )

    train_encoded = apply_id_mappings(
        interactions=(filtered_train),
        user_mapping=(user_mapping),
        item_mapping=(item_mapping),
        user_col=user_col,
        item_col=item_col,
    )

    validation_encoded = apply_id_mappings(
        interactions=validation,
        user_mapping=(user_mapping),
        item_mapping=(item_mapping),
        user_col=user_col,
        item_col=item_col,
        allow_unknown_items=True,
        unknown_index=(unknown_item_index),
    )

    test_encoded = apply_id_mappings(
        interactions=test,
        user_mapping=(user_mapping),
        item_mapping=(item_mapping),
        user_col=user_col,
        item_col=item_col,
        allow_unknown_items=True,
        unknown_index=(unknown_item_index),
    )

    train_encoded = train_encoded.sort(
        "user_idx",
        timestamp_col,
        "item_idx",
    )

    validation_encoded = validation_encoded.sort("user_idx")

    test_encoded = test_encoded.sort("user_idx")

    user_mapping = user_mapping.sort("user_idx")

    item_mapping = item_mapping.sort("item_idx")

    validation_cold_items = validation_encoded.filter(
        pl.col("item_idx") == unknown_item_index
    ).height

    if train_encoded.filter(pl.col("user_idx") < 0).height > 0:
        raise RuntimeError("Training data contains unknown users.")

    if train_encoded.filter(pl.col("item_idx") < 0).height > 0:
        raise RuntimeError("Training data contains unknown items.")

    if validation_encoded.filter(pl.col("user_idx") < 0).height > 0:
        raise RuntimeError("Validation contains unknown users.")

    if test_encoded.filter(pl.col("user_idx") < 0).height > 0:
        raise RuntimeError("Test contains unknown users.")

    expected_users = user_mapping.height

    if validation_encoded.height != expected_users:
        raise RuntimeError(
            "Validation must contain exactly one target per training user."
        )

    if test_encoded.height != expected_users:
        raise RuntimeError("Test must contain exactly one target per training user.")

    if validation_encoded["user_idx"].n_unique() != expected_users:
        raise RuntimeError(
            "Validation does not contain exactly one unique target per user."
        )

    if test_encoded["user_idx"].n_unique() != expected_users:
        raise RuntimeError("Test does not contain exactly one unique target per user.")

    return PreparedTemporalDataset(
        train=train_encoded,
        validation=(validation_encoded),
        test=test_encoded,
        user_mapping=(user_mapping),
        item_mapping=(item_mapping),
        filter_history=(filter_history),
        temporal_diagnostics=(temporal_diagnostics),
        split_validation=(split_validation),
        validation_cold_items=(validation_cold_items),
    )
