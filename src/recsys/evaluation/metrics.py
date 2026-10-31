from __future__ import annotations

import torch


@torch.no_grad()
def leave_one_out_metrics(
    recommendations: torch.Tensor,
    target_items: torch.Tensor,
    ks: tuple[int, ...] = (10, 50),
) -> dict[str, float]:
    """
    Compute Top-K ranking metric sums for leave-one-out evaluation.

    Each user has exactly one relevant target item.

    Cold-item targets are represented by -1. Since valid recommendation
    indices are non-negative, cold targets naturally contribute zero
    to every ranking metric.

    Returned values are metric sums across the batch. The caller is
    responsible for dividing by the total evaluation-user count.
    """

    if recommendations.ndim != 2:
        raise ValueError("recommendations must have shape [batch_size, max_k].")

    if target_items.ndim != 1:
        raise ValueError("target_items must have shape [batch_size].")

    if recommendations.size(0) != target_items.size(0):
        raise ValueError("recommendations and targets must contain the same users.")

    if recommendations.size(0) == 0:
        raise ValueError("recommendations cannot be empty.")

    if recommendations.size(1) == 0:
        raise ValueError("recommendations must contain at least one ranked item.")

    if not ks:
        raise ValueError("ks cannot be empty.")

    if any(not isinstance(k, int) for k in ks):
        raise TypeError("Every evaluation K must be an integer.")

    if any(k <= 0 for k in ks):
        raise ValueError("Every evaluation K must be > 0.")

    max_available_k = recommendations.size(1)

    if max(ks) > max_available_k:
        raise ValueError(
            "Maximum evaluation K cannot exceed "
            "the number of recommendation columns. "
            f"max_k={max(ks)}, "
            f"available={max_available_k}."
        )

    valid_integer_dtypes = {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }

    if recommendations.dtype not in valid_integer_dtypes:
        raise TypeError("recommendations must contain integer item indices.")

    if target_items.dtype not in valid_integer_dtypes:
        raise TypeError("target_items must contain integer item indices.")

    if recommendations.device != target_items.device:
        raise ValueError("recommendations and target_items must be on the same device.")

    if torch.any(recommendations < 0):
        raise ValueError("recommendations cannot contain negative item indices.")

    if torch.any(target_items < -1):
        raise ValueError("target item indices must be >= -1.")

    metrics: dict[
        str,
        float,
    ] = {}

    expanded_targets = target_items.unsqueeze(1)

    for k in ks:
        topk = recommendations[
            :,
            :k,
        ]

        matches = topk.eq(expanded_targets)

        hits = matches.any(dim=1)

        hit_float = hits.float()

        precision = hit_float / k

        recall = hit_float

        ranks = (
            torch.argmax(
                matches.int(),
                dim=1,
            )
            + 1
        )

        reciprocal_rank = torch.where(
            hits,
            1.0 / ranks.float(),
            torch.zeros_like(
                ranks,
                dtype=torch.float32,
            ),
        )

        ndcg = torch.where(
            hits,
            1.0 / torch.log2(ranks.float() + 1.0),
            torch.zeros_like(
                ranks,
                dtype=torch.float32,
            ),
        )

        metrics[f"precision@{k}"] = precision.sum().item()

        metrics[f"recall@{k}"] = recall.sum().item()

        metrics[f"hitrate@{k}"] = recall.sum().item()

        metrics[f"mrr@{k}"] = reciprocal_rank.sum().item()

        metrics[f"ndcg@{k}"] = ndcg.sum().item()

    return metrics


def round_metric_values(
    metrics: dict[str, float],
    decimals: int = 3,
) -> dict[str, float]:
    """
    Round metric values for display without modifying
    raw evaluation results.
    """

    if not isinstance(
        decimals,
        int,
    ):
        raise TypeError("decimals must be an integer.")

    if decimals < 0:
        raise ValueError("decimals must be >= 0.")

    return {
        metric: round(
            value,
            decimals,
        )
        for metric, value in metrics.items()
    }
