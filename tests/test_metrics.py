from __future__ import annotations

import math

import pytest
import torch

from recsys.evaluation.metrics import (
    leave_one_out_metrics,
    round_metric_values,
)


def test_perfect_top_one_hit() -> None:
    recommendations = torch.tensor(
        [
            [3, 5, 7],
        ],
        dtype=torch.long,
    )

    targets = torch.tensor(
        [
            3,
        ],
        dtype=torch.long,
    )

    metrics = leave_one_out_metrics(
        recommendations=recommendations,
        target_items=targets,
        ks=(
            1,
            3,
        ),
    )

    assert metrics["precision@1"] == pytest.approx(1.0)

    assert metrics["recall@1"] == pytest.approx(1.0)

    assert metrics["hitrate@1"] == pytest.approx(1.0)

    assert metrics["mrr@1"] == pytest.approx(1.0)

    assert metrics["ndcg@1"] == pytest.approx(1.0)

    assert metrics["precision@3"] == pytest.approx(1.0 / 3.0)

    assert metrics["recall@3"] == pytest.approx(1.0)

    assert metrics["hitrate@3"] == pytest.approx(1.0)

    assert metrics["mrr@3"] == pytest.approx(1.0)

    assert metrics["ndcg@3"] == pytest.approx(1.0)


def test_hit_at_second_position() -> None:
    recommendations = torch.tensor(
        [
            [10, 20, 30],
        ],
        dtype=torch.long,
    )

    targets = torch.tensor(
        [
            20,
        ],
        dtype=torch.long,
    )

    metrics = leave_one_out_metrics(
        recommendations=recommendations,
        target_items=targets,
        ks=(
            1,
            3,
        ),
    )

    assert metrics["precision@1"] == pytest.approx(0.0)

    assert metrics["recall@1"] == pytest.approx(0.0)

    assert metrics["hitrate@1"] == pytest.approx(0.0)

    assert metrics["mrr@1"] == pytest.approx(0.0)

    assert metrics["ndcg@1"] == pytest.approx(0.0)

    assert metrics["precision@3"] == pytest.approx(1.0 / 3.0)

    assert metrics["recall@3"] == pytest.approx(1.0)

    assert metrics["hitrate@3"] == pytest.approx(1.0)

    assert metrics["mrr@3"] == pytest.approx(0.5)

    assert metrics["ndcg@3"] == pytest.approx(1.0 / math.log2(3.0))


def test_complete_miss() -> None:
    recommendations = torch.tensor(
        [
            [1, 2, 3],
        ],
        dtype=torch.long,
    )

    targets = torch.tensor(
        [
            99,
        ],
        dtype=torch.long,
    )

    metrics = leave_one_out_metrics(
        recommendations=recommendations,
        target_items=targets,
        ks=(
            1,
            3,
        ),
    )

    for value in metrics.values():
        assert value == pytest.approx(0.0)


def test_cold_target_is_valid_and_counts_as_miss() -> None:
    recommendations = torch.tensor(
        [
            [1, 2, 3],
        ],
        dtype=torch.long,
    )

    targets = torch.tensor(
        [
            -1,
        ],
        dtype=torch.long,
    )

    metrics = leave_one_out_metrics(
        recommendations=recommendations,
        target_items=targets,
        ks=(
            1,
            3,
        ),
    )

    for value in metrics.values():
        assert value == pytest.approx(0.0)


def test_batch_metrics_are_returned_as_sums() -> None:
    """
    The metric function returns batch SUMS.

    The evaluator is responsible for dividing by the total
    evaluation-user count after accumulating all batches.
    """

    recommendations = torch.tensor(
        [
            [10, 20, 30],
            [40, 50, 60],
            [70, 80, 90],
        ],
        dtype=torch.long,
    )

    targets = torch.tensor(
        [
            10,
            50,
            -1,
        ],
        dtype=torch.long,
    )

    metrics = leave_one_out_metrics(
        recommendations=recommendations,
        target_items=targets,
        ks=(
            1,
            3,
        ),
    )

    # User 1 hits rank 1.
    # User 2 hits rank 2.
    # User 3 is a cold target and contributes zero.

    assert metrics["precision@1"] == pytest.approx(1.0)

    assert metrics["recall@1"] == pytest.approx(1.0)

    assert metrics["hitrate@1"] == pytest.approx(1.0)

    assert metrics["mrr@1"] == pytest.approx(1.0)

    assert metrics["ndcg@1"] == pytest.approx(1.0)

    assert metrics["precision@3"] == pytest.approx(2.0 / 3.0)

    assert metrics["recall@3"] == pytest.approx(2.0)

    assert metrics["hitrate@3"] == pytest.approx(2.0)

    assert metrics["mrr@3"] == pytest.approx(1.5)

    expected_ndcg = 1.0 + (1.0 / math.log2(3.0))

    assert metrics["ndcg@3"] == pytest.approx(expected_ndcg)


def test_cold_target_remains_in_final_denominator() -> None:
    """
    Two users:
        user 1 -> rank-1 hit
        user 2 -> cold target

    Metric sums should reflect one hit, and dividing by the
    full evaluation population gives 0.5 recall/hitrate.
    """

    recommendations = torch.tensor(
        [
            [10, 20],
            [30, 40],
        ],
        dtype=torch.long,
    )

    targets = torch.tensor(
        [
            10,
            -1,
        ],
        dtype=torch.long,
    )

    metrics = leave_one_out_metrics(
        recommendations=recommendations,
        target_items=targets,
        ks=(1,),
    )

    total_users = targets.numel()

    recall = metrics["recall@1"] / total_users

    hitrate = metrics["hitrate@1"] / total_users

    assert recall == pytest.approx(0.5)

    assert hitrate == pytest.approx(0.5)


def test_known_example_matches_expected_values() -> None:
    """
    Regression test for the cold-target behavior used
    during the preprocessing/evaluator audit.
    """

    recommendations = torch.tensor(
        [
            [10, 20, 30],
            [40, 50, 60],
        ],
        dtype=torch.long,
    )

    targets = torch.tensor(
        [
            20,
            -1,
        ],
        dtype=torch.long,
    )

    metrics = leave_one_out_metrics(
        recommendations=recommendations,
        target_items=targets,
        ks=(
            1,
            3,
        ),
    )

    assert metrics == pytest.approx(
        {
            "precision@1": 0.0,
            "recall@1": 0.0,
            "hitrate@1": 0.0,
            "mrr@1": 0.0,
            "ndcg@1": 0.0,
            "precision@3": (1.0 / 3.0),
            "recall@3": 1.0,
            "hitrate@3": 1.0,
            "mrr@3": 0.5,
            "ndcg@3": (1.0 / math.log2(3.0)),
        }
    )


def test_rejects_target_below_cold_item_sentinel() -> None:
    recommendations = torch.tensor(
        [
            [1, 2, 3],
        ],
        dtype=torch.long,
    )

    targets = torch.tensor(
        [
            -2,
        ],
        dtype=torch.long,
    )

    with pytest.raises(
        ValueError,
    ):
        leave_one_out_metrics(
            recommendations=recommendations,
            target_items=targets,
            ks=(3,),
        )


def test_rejects_negative_recommendation_indices() -> None:
    recommendations = torch.tensor(
        [
            [1, -1, 3],
        ],
        dtype=torch.long,
    )

    targets = torch.tensor(
        [
            1,
        ],
        dtype=torch.long,
    )

    with pytest.raises(
        ValueError,
    ):
        leave_one_out_metrics(
            recommendations=recommendations,
            target_items=targets,
            ks=(3,),
        )


def test_rejects_mismatched_batch_size() -> None:
    recommendations = torch.tensor(
        [
            [1, 2],
            [3, 4],
        ],
        dtype=torch.long,
    )

    targets = torch.tensor(
        [
            1,
        ],
        dtype=torch.long,
    )

    with pytest.raises(
        ValueError,
    ):
        leave_one_out_metrics(
            recommendations=recommendations,
            target_items=targets,
            ks=(1,),
        )


def test_rejects_k_larger_than_recommendation_width() -> None:
    recommendations = torch.tensor(
        [
            [1, 2, 3],
        ],
        dtype=torch.long,
    )

    targets = torch.tensor(
        [
            1,
        ],
        dtype=torch.long,
    )

    with pytest.raises(
        ValueError,
    ):
        leave_one_out_metrics(
            recommendations=recommendations,
            target_items=targets,
            ks=(5,),
        )


def test_rejects_non_positive_k() -> None:
    recommendations = torch.tensor(
        [
            [1, 2, 3],
        ],
        dtype=torch.long,
    )

    targets = torch.tensor(
        [
            1,
        ],
        dtype=torch.long,
    )

    with pytest.raises(
        ValueError,
    ):
        leave_one_out_metrics(
            recommendations=recommendations,
            target_items=targets,
            ks=(0,),
        )


def test_rejects_float_recommendation_tensor() -> None:
    recommendations = torch.tensor(
        [
            [1.0, 2.0],
        ],
        dtype=torch.float32,
    )

    targets = torch.tensor(
        [
            1,
        ],
        dtype=torch.long,
    )

    with pytest.raises(
        TypeError,
    ):
        leave_one_out_metrics(
            recommendations=recommendations,
            target_items=targets,
            ks=(1,),
        )


def test_rejects_float_target_tensor() -> None:
    recommendations = torch.tensor(
        [
            [1, 2],
        ],
        dtype=torch.long,
    )

    targets = torch.tensor(
        [
            1.0,
        ],
        dtype=torch.float32,
    )

    with pytest.raises(
        TypeError,
    ):
        leave_one_out_metrics(
            recommendations=recommendations,
            target_items=targets,
            ks=(1,),
        )


def test_round_metric_values() -> None:
    metrics = {
        "recall@10": 0.123456,
        "ndcg@10": 0.987654,
    }

    rounded = round_metric_values(
        metrics,
        decimals=3,
    )

    assert rounded == {
        "recall@10": 0.123,
        "ndcg@10": 0.988,
    }


def test_round_metric_values_does_not_mutate_input() -> None:
    metrics = {
        "recall@10": 0.123456,
    }

    rounded = round_metric_values(
        metrics,
        decimals=3,
    )

    assert metrics == {
        "recall@10": 0.123456,
    }

    assert rounded == {
        "recall@10": 0.123,
    }


def test_round_metric_values_rejects_negative_decimals() -> None:
    with pytest.raises(
        ValueError,
    ):
        round_metric_values(
            {
                "recall@10": 0.5,
            },
            decimals=-1,
        )
