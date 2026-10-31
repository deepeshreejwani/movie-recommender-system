from __future__ import annotations

import polars as pl
import pytest

from recsys.data.preprocessing import (
    prepare_temporal_dataset,
)


def build_interactions() -> pl.DataFrame:
    """
    Synthetic positive implicit-feedback interactions.

    Users 1-3:
        - have two valid training interactions
        - satisfy a 2-user / 2-item k-core
        - have unique validation/test timestamps

    User 4:
        - has three total interactions
        - but only one training interaction after temporal splitting
        - must therefore be removed by the train-only k-core

    Training catalog after filtering:
        movieId = {10, 11, 12}

    Validation:
        user 1 -> movie 12 : warm
        user 2 -> movie 10 : warm
        user 3 -> movie 99 : cold

    Test:
        user 1 -> movie 98 : cold
        user 2 -> movie 97 : cold
        user 3 -> movie 11 : warm
    """

    return pl.DataFrame(
        {
            "userId": [
                1,
                1,
                1,
                1,
                2,
                2,
                2,
                2,
                3,
                3,
                3,
                3,
                4,
                4,
                4,
            ],
            "movieId": [
                10,
                11,
                12,
                98,
                11,
                12,
                10,
                97,
                12,
                10,
                99,
                11,
                10,
                11,
                12,
            ],
            "timestamp": [
                10,
                20,
                30,
                40,
                10,
                20,
                30,
                40,
                10,
                20,
                30,
                40,
                10,
                20,
                30,
            ],
        }
    )


def prepare():
    return prepare_temporal_dataset(
        build_interactions(),
        min_user_interactions=2,
        min_item_interactions=2,
        unknown_item_index=-1,
    )


def test_train_only_k_core_removes_user_with_insufficient_train_history() -> None:
    prepared = prepare()

    train_users = (
        prepared.train.select("userId").unique().sort("userId").to_series().to_list()
    )

    assert train_users == [
        1,
        2,
        3,
    ]

    assert 4 not in train_users

    # User 4 had three total interactions, so filtering the
    # complete dataset before splitting would incorrectly
    # allow future holdouts to contribute to eligibility.
    assert build_interactions().filter(pl.col("userId") == 4).height == 3


def test_train_only_k_core_thresholds_are_satisfied() -> None:
    prepared = prepare()

    user_counts = prepared.train.group_by("user_idx").len()

    item_counts = prepared.train.group_by("item_idx").len()

    assert user_counts["len"].min() >= 2

    assert item_counts["len"].min() >= 2


def test_mappings_are_built_from_filtered_training_only() -> None:
    prepared = prepare()

    mapped_users = prepared.user_mapping.sort("user_idx")

    mapped_items = prepared.item_mapping.sort("item_idx")

    assert mapped_users["userId"].to_list() == [
        1,
        2,
        3,
    ]

    assert mapped_users["user_idx"].to_list() == [
        0,
        1,
        2,
    ]

    assert mapped_items["movieId"].to_list() == [
        10,
        11,
        12,
    ]

    assert mapped_items["item_idx"].to_list() == [
        0,
        1,
        2,
    ]

    # Holdout-only items must not expand the training catalog.
    assert 99 not in (mapped_items["movieId"].to_list())

    assert 97 not in (mapped_items["movieId"].to_list())

    assert 98 not in (mapped_items["movieId"].to_list())


def test_cold_validation_item_is_preserved_as_negative_one() -> None:
    prepared = prepare()

    validation = prepared.validation.sort("userId")

    assert validation.height == 3

    assert validation["userId"].to_list() == [
        1,
        2,
        3,
    ]

    assert validation["movieId"].to_list() == [
        12,
        10,
        99,
    ]

    assert validation["item_idx"].to_list() == [
        2,
        0,
        -1,
    ]

    assert prepared.validation_cold_items == 1


def test_cold_test_items_are_preserved_as_evaluation_misses() -> None:
    prepared = prepare()

    test = prepared.test.sort("userId")

    assert test.height == 3

    assert test["movieId"].to_list() == [
        98,
        97,
        11,
    ]

    assert test["item_idx"].to_list() == [
        -1,
        -1,
        1,
    ]

    # Test rows must remain present even when targets
    # are outside the training catalog.
    assert test["user_idx"].n_unique() == 3


def test_holdout_users_are_exactly_the_filtered_training_users() -> None:
    prepared = prepare()

    train_users = set(prepared.train["user_idx"].unique())

    validation_users = set(prepared.validation["user_idx"])

    test_users = set(prepared.test["user_idx"])

    assert train_users == validation_users == test_users

    assert len(train_users) == 3


def test_test_item_identity_does_not_affect_preprocessing() -> None:
    """
    Changing only a test target to an item already consumed
    in that user's training history must not affect:

    - training eligibility
    - training interactions
    - validation interactions
    - user vocabulary
    - item vocabulary
    - preprocessing success

    The repeated train/test pair is diagnostic only.

    At downstream evaluation the repeated target will be
    masked as already seen and therefore contribute a miss.
    """

    original = build_interactions()

    # User 1 training history contains movie 10.
    #
    # Replace only the user's test target:
    #
    #     movie 98 -> movie 10
    #
    # Timestamp and every other interaction remain unchanged.
    modified = original.with_columns(
        pl.when((pl.col("userId") == 1) & (pl.col("timestamp") == 40))
        .then(pl.lit(10))
        .otherwise(pl.col("movieId"))
        .alias("movieId")
    )

    prepared_original = prepare_temporal_dataset(
        original,
        min_user_interactions=2,
        min_item_interactions=2,
        unknown_item_index=-1,
    )

    prepared_modified = prepare_temporal_dataset(
        modified,
        min_user_interactions=2,
        min_item_interactions=2,
        unknown_item_index=-1,
    )

    # -----------------------------------------------------
    # Training partition must be completely unchanged.
    # -----------------------------------------------------

    assert prepared_original.train.sort(
        [
            "user_idx",
            "timestamp",
            "item_idx",
        ]
    ).equals(
        prepared_modified.train.sort(
            [
                "user_idx",
                "timestamp",
                "item_idx",
            ]
        )
    )

    # -----------------------------------------------------
    # Validation partition must be completely unchanged.
    # -----------------------------------------------------

    assert prepared_original.validation.sort("user_idx").equals(
        prepared_modified.validation.sort("user_idx")
    )

    # -----------------------------------------------------
    # Training-derived vocabularies must be unchanged.
    # -----------------------------------------------------

    assert prepared_original.user_mapping.sort("user_idx").equals(
        prepared_modified.user_mapping.sort("user_idx")
    )

    assert prepared_original.item_mapping.sort("item_idx").equals(
        prepared_modified.item_mapping.sort("item_idx")
    )

    # -----------------------------------------------------
    # User cohort must remain unchanged.
    # -----------------------------------------------------

    assert (
        prepared_original.train["user_idx"].n_unique()
        == prepared_modified.train["user_idx"].n_unique()
        == 3
    )

    # -----------------------------------------------------
    # Only the intended test target changes.
    #
    # Original movie 98 is outside the training catalog -> -1.
    # Modified movie 10 is already in the catalog -> item_idx 0.
    # -----------------------------------------------------

    original_user_one_test = prepared_original.test.filter(pl.col("userId") == 1)

    modified_user_one_test = prepared_modified.test.filter(pl.col("userId") == 1)

    assert original_user_one_test["movieId"].item() == 98

    assert original_user_one_test["item_idx"].item() == -1

    assert modified_user_one_test["movieId"].item() == 10

    assert modified_user_one_test["item_idx"].item() == 0

    # -----------------------------------------------------
    # Pair overlap is visible as a diagnostic but does not
    # invalidate the preprocessing run.
    # -----------------------------------------------------

    assert prepared_modified.split_validation["train_test_pair_overlap"] == 1

    assert prepared_modified.split_validation["pair_overlap_free"] is False

    assert prepared_modified.split_validation["valid"] is True


def test_unknown_item_index_must_be_negative_one() -> None:
    with pytest.raises(
        ValueError,
    ):
        prepare_temporal_dataset(
            build_interactions(),
            min_user_interactions=2,
            min_item_interactions=2,
            unknown_item_index=0,
        )
