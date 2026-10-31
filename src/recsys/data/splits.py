from __future__ import annotations

from typing import Literal, overload

import polars as pl


def _validate_columns(
    interactions: pl.DataFrame,
    required_columns: set[str],
    frame_name: str,
) -> None:
    """Validate required columns for a dataframe."""

    missing_columns = required_columns - set(interactions.columns)

    if missing_columns:
        raise ValueError(
            f"{frame_name} is missing required columns: {sorted(missing_columns)}"
        )


def _validate_non_null_columns(
    interactions: pl.DataFrame,
    columns: tuple[str, ...],
    frame_name: str,
) -> None:
    """Reject null values in required split columns."""

    for column in columns:
        null_count = interactions[column].null_count()

        if null_count > 0:
            raise ValueError(
                f"{frame_name} contains {null_count} null values in '{column}'."
            )


def _build_temporal_boundaries(
    interactions: pl.DataFrame,
    user_col: str,
    timestamp_col: str,
) -> pl.DataFrame:
    """
    Build strict per-user validation/test boundaries.

    Requirements for an eligible user:

    - at least 3 distinct timestamps
    - latest timestamp contains exactly one interaction
    - second-latest timestamp contains exactly one interaction
    """

    timestamp_groups = (
        interactions.group_by(
            [
                user_col,
                timestamp_col,
            ]
        )
        .len()
        .rename(
            {
                "len": ("_timestamp_count"),
            }
        )
        .with_columns(
            pl.col(timestamp_col)
            .rank(
                method="dense",
                descending=True,
            )
            .over(user_col)
            .alias("_timestamp_rank")
        )
    )

    boundaries = timestamp_groups.group_by(user_col).agg(
        pl.len().alias("_n_distinct_timestamps"),
        pl.col(timestamp_col)
        .filter(pl.col("_timestamp_rank") == 1)
        .first()
        .alias("_test_timestamp"),
        pl.col("_timestamp_count")
        .filter(pl.col("_timestamp_rank") == 1)
        .first()
        .alias("_test_timestamp_count"),
        pl.col(timestamp_col)
        .filter(pl.col("_timestamp_rank") == 2)
        .first()
        .alias("_validation_timestamp"),
        pl.col("_timestamp_count")
        .filter(pl.col("_timestamp_rank") == 2)
        .first()
        .alias("_validation_timestamp_count"),
    )

    return boundaries


def temporal_split_diagnostics(
    interactions: pl.DataFrame,
    user_col: str = "userId",
    timestamp_col: str = "timestamp",
) -> dict[
    str,
    int | float,
]:
    """
    Summarize user eligibility for strict temporal
    leave-two-out splitting.

    Diagnostic exclusion categories may overlap.
    """

    if interactions.is_empty():
        raise ValueError("interactions cannot be empty.")

    required_columns = {
        user_col,
        timestamp_col,
    }

    _validate_columns(
        interactions=interactions,
        required_columns=(required_columns),
        frame_name="interactions",
    )

    _validate_non_null_columns(
        interactions=interactions,
        columns=(
            user_col,
            timestamp_col,
        ),
        frame_name="interactions",
    )

    boundaries = _build_temporal_boundaries(
        interactions=interactions,
        user_col=user_col,
        timestamp_col=(timestamp_col),
    )

    total_users = boundaries.height

    insufficient_timestamps = boundaries.filter(
        pl.col("_n_distinct_timestamps") < 3
    ).height

    latest_timestamp_ties = boundaries.filter(
        pl.col("_test_timestamp_count") > 1
    ).height

    validation_timestamp_ties = boundaries.filter(
        pl.col("_validation_timestamp_count") > 1
    ).height

    eligible_users = boundaries.filter(
        (pl.col("_n_distinct_timestamps") >= 3)
        & (pl.col("_test_timestamp_count") == 1)
        & (pl.col("_validation_timestamp_count") == 1)
    ).height

    eligible_user_pct = eligible_users / total_users * 100 if total_users else 0.0

    return {
        "total_users": (total_users),
        "eligible_users": (eligible_users),
        "excluded_users": (total_users - eligible_users),
        "insufficient_distinct_timestamps": (insufficient_timestamps),
        "latest_timestamp_ties": (latest_timestamp_ties),
        "validation_timestamp_ties": (validation_timestamp_ties),
        "eligible_user_pct": (eligible_user_pct),
    }


@overload
def temporal_user_split(
    interactions: pl.DataFrame,
    user_col: str = "userId",
    item_col: str = "movieId",
    timestamp_col: str = "timestamp",
    return_diagnostics: Literal[False] = False,
) -> tuple[
    pl.DataFrame,
    pl.DataFrame,
    pl.DataFrame,
]: ...


@overload
def temporal_user_split(
    interactions: pl.DataFrame,
    user_col: str = "userId",
    item_col: str = "movieId",
    timestamp_col: str = "timestamp",
    *,
    return_diagnostics: Literal[True],
) -> tuple[
    pl.DataFrame,
    pl.DataFrame,
    pl.DataFrame,
    dict[
        str,
        int | float,
    ],
]: ...


@overload
def temporal_user_split(
    interactions: pl.DataFrame,
    user_col: str,
    item_col: str,
    timestamp_col: str,
    return_diagnostics: bool,
) -> (
    tuple[
        pl.DataFrame,
        pl.DataFrame,
        pl.DataFrame,
    ]
    | tuple[
        pl.DataFrame,
        pl.DataFrame,
        pl.DataFrame,
        dict[
            str,
            int | float,
        ],
    ]
): ...


def temporal_user_split(
    interactions: pl.DataFrame,
    user_col: str = "userId",
    item_col: str = "movieId",
    timestamp_col: str = "timestamp",
    return_diagnostics: bool = False,
) -> (
    tuple[
        pl.DataFrame,
        pl.DataFrame,
        pl.DataFrame,
    ]
    | tuple[
        pl.DataFrame,
        pl.DataFrame,
        pl.DataFrame,
        dict[
            str,
            int | float,
        ],
    ]
):
    """
    Strict per-user temporal leave-two-out split.

    Users are retained only when:

    - at least 3 distinct timestamps exist
    - latest timestamp has exactly one interaction
    - second-latest timestamp has exactly one interaction

    For each retained user:

    - train: timestamp < validation timestamp
    - validation: unique second-latest timestamp
    - test: unique latest timestamp

    User-item identity does not participate in user
    eligibility or temporal-boundary selection.
    """

    if interactions.is_empty():
        raise ValueError("interactions cannot be empty.")

    required_columns = {
        user_col,
        item_col,
        timestamp_col,
    }

    _validate_columns(
        interactions=interactions,
        required_columns=(required_columns),
        frame_name="interactions",
    )

    _validate_non_null_columns(
        interactions=interactions,
        columns=(
            user_col,
            item_col,
            timestamp_col,
        ),
        frame_name="interactions",
    )

    original_columns = interactions.columns

    boundaries = _build_temporal_boundaries(
        interactions=interactions,
        user_col=user_col,
        timestamp_col=(timestamp_col),
    )

    eligible_boundaries = boundaries.filter(
        (pl.col("_n_distinct_timestamps") >= 3)
        & (pl.col("_test_timestamp_count") == 1)
        & (pl.col("_validation_timestamp_count") == 1)
    )

    if eligible_boundaries.is_empty():
        raise ValueError("No users satisfy the strict temporal split requirements.")

    split_data = interactions.join(
        eligible_boundaries.select(
            [
                user_col,
                ("_validation_timestamp"),
                "_test_timestamp",
            ]
        ),
        on=user_col,
        how="inner",
    )

    train = split_data.filter(
        pl.col(timestamp_col) < pl.col("_validation_timestamp")
    ).select(original_columns)

    validation = split_data.filter(
        pl.col(timestamp_col) == pl.col("_validation_timestamp")
    ).select(original_columns)

    test = split_data.filter(pl.col(timestamp_col) == pl.col("_test_timestamp")).select(
        original_columns
    )

    if train.is_empty() or validation.is_empty() or test.is_empty():
        raise RuntimeError("Temporal splitting produced an empty partition.")

    if return_diagnostics:
        diagnostics = temporal_split_diagnostics(
            interactions=(interactions),
            user_col=user_col,
            timestamp_col=(timestamp_col),
        )

        return (
            train,
            validation,
            test,
            diagnostics,
        )

    return (
        train,
        validation,
        test,
    )


def _pair_overlap_count(
    left: pl.DataFrame,
    right: pl.DataFrame,
    user_col: str,
    item_col: str,
) -> int:
    """
    Count unique user-item pairs occurring
    in both partitions.

    Pair overlap is diagnostic only. Repeated consumption
    of the same item at different timestamps is valid
    temporal data and must not alter split eligibility.
    """

    return (
        left.select(
            [
                user_col,
                item_col,
            ]
        )
        .unique()
        .join(
            right.select(
                [
                    user_col,
                    item_col,
                ]
            ).unique(),
            on=[
                user_col,
                item_col,
            ],
            how="inner",
        )
        .height
    )


def _same_user_set(
    left: pl.DataFrame,
    right: pl.DataFrame,
    user_col: str,
) -> bool:
    """Check whether two partitions contain identical users."""

    left_users = left.select(user_col).unique().sort(user_col)

    right_users = right.select(user_col).unique().sort(user_col)

    return left_users.equals(right_users)


def validate_temporal_split(
    train: pl.DataFrame,
    validation: pl.DataFrame,
    test: pl.DataFrame,
    user_col: str = "userId",
    item_col: str = "movieId",
    timestamp_col: str = "timestamp",
) -> dict[
    str,
    int | bool,
]:
    """
    Validate strict temporal ordering and cohort consistency.

    Required invariants:

    - one validation target per user
    - one test target per user
    - train/validation/test contain the same users
    - train timestamps strictly precede validation
    - validation timestamps strictly precede test

    User-item pair overlap is intentionally diagnostic only.

    A validation or test target may refer to an item the user
    consumed previously. Target identity must not affect
    preprocessing eligibility or whether a temporal split is
    considered valid. Downstream seen-item masking naturally
    makes such repeated targets evaluation misses.
    """

    required_columns = {
        user_col,
        item_col,
        timestamp_col,
    }

    frames = {
        "train": train,
        "validation": (validation),
        "test": test,
    }

    for (
        frame_name,
        frame,
    ) in frames.items():
        if frame.is_empty():
            raise ValueError(f"{frame_name} cannot be empty.")

        _validate_columns(
            interactions=frame,
            required_columns=(required_columns),
            frame_name=(frame_name),
        )

        _validate_non_null_columns(
            interactions=frame,
            columns=(
                user_col,
                item_col,
                timestamp_col,
            ),
            frame_name=(frame_name),
        )

    train_users = train[user_col].n_unique()

    validation_users = validation[user_col].n_unique()

    test_users = test[user_col].n_unique()

    validation_one_per_user = validation.height == validation_users

    test_one_per_user = test.height == test_users

    same_train_validation_users = _same_user_set(
        left=train,
        right=validation,
        user_col=user_col,
    )

    same_validation_test_users = _same_user_set(
        left=validation,
        right=test,
        user_col=user_col,
    )

    same_evaluation_users = same_train_validation_users and same_validation_test_users

    # -----------------------------------------------------
    # Pair-overlap diagnostics
    #
    # These values are informative only and deliberately
    # do not determine split validity.
    # -----------------------------------------------------

    train_validation_overlap = _pair_overlap_count(
        left=train,
        right=validation,
        user_col=user_col,
        item_col=item_col,
    )

    train_test_overlap = _pair_overlap_count(
        left=train,
        right=test,
        user_col=user_col,
        item_col=item_col,
    )

    validation_test_overlap = _pair_overlap_count(
        left=validation,
        right=test,
        user_col=user_col,
        item_col=item_col,
    )

    pair_overlap_free = (
        train_validation_overlap == 0
        and train_test_overlap == 0
        and validation_test_overlap == 0
    )

    # -----------------------------------------------------
    # Strict temporal boundaries
    # -----------------------------------------------------

    validation_boundaries = validation.select(
        [
            user_col,
            pl.col(timestamp_col).alias("_validation_timestamp"),
        ]
    )

    test_boundaries = test.select(
        [
            user_col,
            pl.col(timestamp_col).alias("_test_timestamp"),
        ]
    )

    train_max_timestamp = train.group_by(user_col).agg(
        pl.col(timestamp_col).max().alias("_train_max_timestamp")
    )

    temporal_boundaries = validation_boundaries.join(
        test_boundaries,
        on=user_col,
        how="inner",
    ).join(
        train_max_timestamp,
        on=user_col,
        how="left",
    )

    missing_train_boundary = temporal_boundaries["_train_max_timestamp"].null_count()

    train_before_validation = missing_train_boundary == 0 and (
        temporal_boundaries.filter(
            pl.col("_train_max_timestamp") >= pl.col("_validation_timestamp")
        ).height
        == 0
    )

    validation_before_test = (
        temporal_boundaries.filter(
            pl.col("_validation_timestamp") >= pl.col("_test_timestamp")
        ).height
        == 0
    )

    result: dict[
        str,
        int | bool,
    ] = {
        "train_users": (train_users),
        "validation_users": (validation_users),
        "test_users": (test_users),
        "validation_one_per_user": (validation_one_per_user),
        "test_one_per_user": (test_one_per_user),
        "same_train_validation_users": (same_train_validation_users),
        "same_validation_test_users": (same_validation_test_users),
        "same_evaluation_users": (same_evaluation_users),
        "train_validation_pair_overlap": (train_validation_overlap),
        "train_test_pair_overlap": (train_test_overlap),
        "validation_test_pair_overlap": (validation_test_overlap),
        "pair_overlap_free": (pair_overlap_free),
        "strict_train_before_validation": (train_before_validation),
        "strict_validation_before_test": (validation_before_test),
    }

    valid = (
        validation_one_per_user
        and test_one_per_user
        and same_evaluation_users
        and train_before_validation
        and validation_before_test
    )

    result["valid"] = valid

    if not valid:
        raise ValueError(f"Temporal split validation failed: {result}")

    return result
