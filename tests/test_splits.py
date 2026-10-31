from __future__ import annotations

import polars as pl
import pytest

from recsys.data.splits import (
    temporal_split_diagnostics,
    temporal_user_split,
    validate_temporal_split,
)


def build_interactions() -> pl.DataFrame:
    """
    Synthetic interactions covering the important temporal cases.

    Valid users:
        user 1 -> unique latest + validation timestamps
        user 2 -> exactly three distinct timestamps
        user 6 -> ties are allowed inside the training history

    Excluded users:
        user 3 -> fewer than three distinct timestamps
        user 4 -> latest timestamp contains a tie
        user 5 -> second-latest timestamp contains a tie
    """

    return pl.DataFrame(
        {
            "userId": [
                # User 1: valid
                1,
                1,
                1,
                1,
                # User 2: valid
                2,
                2,
                2,
                # User 3: insufficient distinct timestamps
                3,
                3,
                3,
                # User 4: latest timestamp tie
                4,
                4,
                4,
                4,
                # User 5: validation timestamp tie
                5,
                5,
                5,
                5,
                # User 6: valid, earlier train timestamp tie
                6,
                6,
                6,
                6,
                6,
            ],
            "movieId": [
                # User 1
                10,
                11,
                12,
                13,
                # User 2
                20,
                21,
                22,
                # User 3
                30,
                31,
                32,
                # User 4
                40,
                41,
                42,
                43,
                # User 5
                50,
                51,
                52,
                53,
                # User 6
                60,
                61,
                62,
                63,
                64,
            ],
            "timestamp": [
                # User 1
                10,
                20,
                30,
                40,
                # User 2
                10,
                20,
                30,
                # User 3
                10,
                20,
                20,
                # User 4
                10,
                20,
                30,
                30,
                # User 5
                10,
                20,
                20,
                30,
                # User 6
                10,
                10,
                20,
                30,
                40,
            ],
        }
    )


def test_temporal_split_keeps_only_eligible_users() -> None:
    train, validation, test = temporal_user_split(build_interactions())

    expected_users = {
        1,
        2,
        6,
    }

    assert set(train["userId"].unique()) == expected_users

    assert set(validation["userId"]) == expected_users

    assert set(test["userId"]) == expected_users


def test_temporal_split_has_exactly_one_validation_and_test_row_per_user() -> None:
    _, validation, test = temporal_user_split(build_interactions())

    assert validation.height == 3
    assert test.height == 3

    assert validation["userId"].n_unique() == 3

    assert test["userId"].n_unique() == 3

    validation_counts = validation.group_by("userId").len()

    test_counts = test.group_by("userId").len()

    assert validation_counts["len"].max() == 1

    assert test_counts["len"].max() == 1


def test_temporal_split_uses_strict_timestamp_ordering() -> None:
    train, validation, test = temporal_user_split(build_interactions())

    train_max = train.group_by("userId").agg(
        pl.col("timestamp").max().alias("train_timestamp")
    )

    validation_times = validation.select(
        "userId",
        pl.col("timestamp").alias("validation_timestamp"),
    )

    test_times = test.select(
        "userId",
        pl.col("timestamp").alias("test_timestamp"),
    )

    boundaries = train_max.join(
        validation_times,
        on="userId",
        how="inner",
    ).join(
        test_times,
        on="userId",
        how="inner",
    )

    assert boundaries.filter(
        ~(
            (pl.col("train_timestamp") < pl.col("validation_timestamp"))
            & (pl.col("validation_timestamp") < pl.col("test_timestamp"))
        )
    ).is_empty()


def test_latest_timestamp_ties_are_excluded() -> None:
    train, validation, test = temporal_user_split(build_interactions())

    for frame in (
        train,
        validation,
        test,
    ):
        assert frame.filter(pl.col("userId") == 4).is_empty()


def test_validation_timestamp_ties_are_excluded() -> None:
    train, validation, test = temporal_user_split(build_interactions())

    for frame in (
        train,
        validation,
        test,
    ):
        assert frame.filter(pl.col("userId") == 5).is_empty()


def test_training_history_may_contain_earlier_timestamp_ties() -> None:
    train, validation, test = temporal_user_split(build_interactions())

    user_six_train = train.filter(pl.col("userId") == 6).sort(
        [
            "timestamp",
            "movieId",
        ]
    )

    assert user_six_train.height == 3

    assert user_six_train["timestamp"].to_list() == [
        10,
        10,
        20,
    ]

    assert validation.filter(pl.col("userId") == 6)["timestamp"].item() == 30

    assert test.filter(pl.col("userId") == 6)["timestamp"].item() == 40


def test_temporal_split_diagnostics() -> None:
    diagnostics = temporal_split_diagnostics(build_interactions())

    assert diagnostics["total_users"] == 6

    assert diagnostics["eligible_users"] == 3

    assert diagnostics["excluded_users"] == 3

    assert diagnostics["insufficient_distinct_timestamps"] == 1

    # Diagnostic exclusion reasons can overlap.
    # User 3 has insufficient distinct timestamps and
    # also has a tie at the latest timestamp.
    assert diagnostics["latest_timestamp_ties"] == 2

    assert diagnostics["validation_timestamp_ties"] == 1

    assert diagnostics["eligible_user_pct"] == pytest.approx(50.0)


def test_return_diagnostics_matches_direct_diagnostics() -> None:
    (
        train,
        validation,
        test,
        diagnostics,
    ) = temporal_user_split(
        build_interactions(),
        return_diagnostics=True,
    )

    expected = temporal_split_diagnostics(build_interactions())

    assert diagnostics == expected

    assert not train.is_empty()
    assert not validation.is_empty()
    assert not test.is_empty()


def test_temporal_split_is_independent_of_input_row_order() -> None:
    interactions = build_interactions()

    reversed_interactions = interactions.reverse()

    original = temporal_user_split(interactions)

    reversed_result = temporal_user_split(reversed_interactions)

    for (
        original_frame,
        reversed_frame,
    ) in zip(
        original,
        reversed_result,
        strict=True,
    ):
        sort_columns = [
            "userId",
            "timestamp",
            "movieId",
        ]

        assert original_frame.sort(sort_columns).equals(
            reversed_frame.sort(sort_columns)
        )


def test_validate_temporal_split_accepts_valid_split() -> None:
    train, validation, test = temporal_user_split(build_interactions())

    result = validate_temporal_split(
        train,
        validation,
        test,
    )

    assert isinstance(
        result,
        dict,
    )

    assert result["valid"] is True


def test_validate_temporal_split_rejects_user_set_mismatch() -> None:
    train, validation, test = temporal_user_split(build_interactions())

    validation = validation.filter(pl.col("userId") != 1)

    with pytest.raises(
        ValueError,
    ):
        validate_temporal_split(
            train,
            validation,
            test,
        )


def test_validate_temporal_split_rejects_non_strict_boundary() -> None:
    train = pl.DataFrame(
        {
            "userId": [
                1,
            ],
            "movieId": [
                10,
            ],
            "timestamp": [
                20,
            ],
        }
    )

    validation = pl.DataFrame(
        {
            "userId": [
                1,
            ],
            "movieId": [
                11,
            ],
            "timestamp": [
                20,
            ],
        }
    )

    test = pl.DataFrame(
        {
            "userId": [
                1,
            ],
            "movieId": [
                12,
            ],
            "timestamp": [
                30,
            ],
        }
    )

    with pytest.raises(
        ValueError,
    ):
        validate_temporal_split(
            train,
            validation,
            test,
        )


def test_validate_temporal_split_reports_pair_overlap_without_rejecting() -> None:
    """
    Repeated user-item pairs across temporal partitions are
    diagnostic only.

    Changing the identity of a test target to an item already
    consumed in training must not invalidate the temporal split.
    """

    train = pl.DataFrame(
        {
            "userId": [
                1,
                1,
            ],
            "movieId": [
                10,
                20,
            ],
            "timestamp": [
                10,
                20,
            ],
        }
    )

    validation = pl.DataFrame(
        {
            "userId": [
                1,
            ],
            "movieId": [
                30,
            ],
            "timestamp": [
                30,
            ],
        }
    )

    # Test target repeats movie 10 from training.
    test = pl.DataFrame(
        {
            "userId": [
                1,
            ],
            "movieId": [
                10,
            ],
            "timestamp": [
                40,
            ],
        }
    )

    result = validate_temporal_split(
        train,
        validation,
        test,
    )

    assert result["valid"] is True

    assert result["strict_train_before_validation"] is True

    assert result["strict_validation_before_test"] is True

    assert result["train_validation_pair_overlap"] == 0

    assert result["train_test_pair_overlap"] == 1

    assert result["validation_test_pair_overlap"] == 0

    assert result["pair_overlap_free"] is False


def test_temporal_split_rejects_null_timestamps() -> None:
    interactions = build_interactions().with_columns(
        pl.when(pl.col("userId") == 1)
        .then(pl.lit(None))
        .otherwise(pl.col("timestamp"))
        .alias("timestamp")
    )

    with pytest.raises(
        ValueError,
    ):
        temporal_user_split(interactions)


def test_temporal_split_rejects_empty_input() -> None:
    interactions = pl.DataFrame(
        schema={
            "userId": pl.Int64,
            "movieId": pl.Int64,
            "timestamp": pl.Int64,
        }
    )

    with pytest.raises(
        ValueError,
    ):
        temporal_user_split(interactions)
