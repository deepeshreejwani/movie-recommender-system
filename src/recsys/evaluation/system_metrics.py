from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch import nn


def count_model_parameters(
    model: nn.Module,
) -> dict[str, int]:
    """Count total and trainable model parameters."""

    total_parameters = sum(parameter.numel() for parameter in model.parameters())

    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )

    return {
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
    }


def artifact_size_mb(
    path: str | Path,
) -> float:
    """Return artifact size in MB."""

    path = Path(path)

    return path.stat().st_size / (1024**2)


def embedding_memory_mb(
    n_entities: int,
    embedding_dim: int,
    bytes_per_value: int = 4,
) -> float:
    """Estimate embedding-table memory requirement."""

    return n_entities * embedding_dim * bytes_per_value / (1024**2)


def embedding_scaling_table(
    entity_counts: tuple[int, ...],
    embedding_dims: tuple[int, ...],
    bytes_per_value: int = 4,
) -> pl.DataFrame:
    """Generate embedding-memory projections."""

    rows = []

    for entity_count in entity_counts:
        for embedding_dim in embedding_dims:
            rows.append(
                {
                    "entities": entity_count,
                    "embedding_dim": embedding_dim,
                    "memory_mb": embedding_memory_mb(
                        n_entities=entity_count,
                        embedding_dim=embedding_dim,
                        bytes_per_value=bytes_per_value,
                    ),
                }
            )

    return pl.DataFrame(rows).with_columns(pl.col("memory_mb").round(3))


def tensor_memory_mb(
    tensor: torch.Tensor,
) -> float:
    """Return in-memory tensor size in MB."""

    return tensor.numel() * tensor.element_size() / (1024**2)


def _validate_recommendation_indices(
    recommendations: np.ndarray,
    n_items: int,
) -> np.ndarray:
    """
    Validate recommendation item indices against the catalog.

    Evaluator padding IDs such as ``item_idx == n_items`` must never
    enter catalog-coverage or popularity metrics.
    """

    if n_items <= 0:
        raise ValueError("n_items must be positive.")

    recommendations = np.asarray(recommendations)

    if recommendations.size == 0:
        raise ValueError("recommendations cannot be empty.")

    if not np.issubdtype(
        recommendations.dtype,
        np.integer,
    ):
        raise TypeError("recommendations must contain integer item indices.")

    flattened = recommendations.reshape(-1).astype(
        np.int64,
        copy=False,
    )

    if np.any(flattened < 0):
        raise ValueError("recommendations contain negative item indices.")

    if np.any(flattened >= n_items):
        raise ValueError("recommendations contain item indices outside the catalog.")

    return flattened


def popularity_concentration(
    interactions: pl.DataFrame,
    item_col: str = "item_idx",
    top_percentages: tuple[float, ...] = (
        0.01,
        0.05,
        0.10,
        0.20,
    ),
) -> pl.DataFrame:
    """Measure interaction concentration among the most popular items."""

    if interactions.is_empty():
        raise ValueError("interactions cannot be empty.")

    if item_col not in interactions.columns:
        raise ValueError(f"interactions is missing column '{item_col}'.")

    popularity = (
        interactions.group_by(item_col)
        .len()
        .rename(
            {
                "len": "interactions",
            }
        )
        .sort(
            "interactions",
            descending=True,
        )
    )

    total_items = popularity.height

    total_interactions = int(popularity["interactions"].sum())

    if total_items <= 0:
        raise RuntimeError("Popularity table contains no items.")

    if total_interactions <= 0:
        raise RuntimeError("Popularity table contains no interactions.")

    rows = []

    for fraction in top_percentages:
        if fraction <= 0 or fraction > 1:
            raise ValueError("top_percentages values must be in the interval (0, 1].")

        top_n = max(
            1,
            int(np.ceil(total_items * fraction)),
        )

        top_interactions = int(popularity.head(top_n)["interactions"].sum())

        rows.append(
            {
                "top_catalog_pct": (fraction * 100),
                "items": top_n,
                "interaction_share_pct": (top_interactions / total_interactions * 100),
            }
        )

    return pl.DataFrame(rows).with_columns(pl.exclude("items").round(3))


def recommendation_catalog_metrics(
    recommendations: np.ndarray,
    n_items: int,
    head_items: np.ndarray | None = None,
) -> dict[str, float | int]:
    """Measure catalog coverage and head/tail recommendation exposure."""

    flattened = _validate_recommendation_indices(
        recommendations=recommendations,
        n_items=n_items,
    )

    unique_items = np.unique(flattened)

    metrics: dict[
        str,
        float | int,
    ] = {
        "recommendations": int(flattened.size),
        "unique_items_recommended": int(unique_items.size),
        "catalog_coverage_pct": float(unique_items.size / n_items * 100),
    }

    if head_items is not None:
        head_items = np.asarray(head_items)

        if not np.issubdtype(
            head_items.dtype,
            np.integer,
        ):
            raise TypeError("head_items must contain integer item indices.")

        head_items = head_items.astype(
            np.int64,
            copy=False,
        )

        if np.any(head_items < 0):
            raise ValueError("head_items contains negative item indices.")

        if np.any(head_items >= n_items):
            raise ValueError("head_items contains item indices outside the catalog.")

        head_mask = np.isin(
            flattened,
            head_items,
        )

        unique_head_items = np.intersect1d(
            unique_items,
            head_items,
        )

        unique_tail_items = np.setdiff1d(
            unique_items,
            head_items,
        )

        metrics.update(
            {
                "head_recommendation_share_pct": float(head_mask.mean() * 100),
                "tail_recommendation_share_pct": float((~head_mask).mean() * 100),
                "unique_head_items": int(unique_head_items.size),
                "unique_tail_items": int(unique_tail_items.size),
            }
        )

    return metrics


def recommendation_popularity_metrics(
    recommendations: np.ndarray,
    item_popularity: pl.DataFrame,
    total_interactions: int,
    item_col: str = "item_idx",
    count_col: str = "interaction_count",
) -> dict[str, float]:
    """
    Measure recommendation popularity and self-information novelty.

    Novelty:
        -log2(P(item))

    where P(item) is estimated from historical interaction frequency.
    """

    if item_popularity.is_empty():
        raise ValueError("item_popularity cannot be empty.")

    required_columns = {
        item_col,
        count_col,
    }

    missing_columns = required_columns - set(item_popularity.columns)

    if missing_columns:
        raise ValueError(
            f"item_popularity is missing columns: {sorted(missing_columns)}"
        )

    if total_interactions <= 0:
        raise ValueError("total_interactions must be positive.")

    if item_popularity[item_col].null_count() > 0:
        raise ValueError(f"item_popularity contains null values in '{item_col}'.")

    if item_popularity[count_col].null_count() > 0:
        raise ValueError(f"item_popularity contains null values in '{count_col}'.")

    if not item_popularity[item_col].dtype.is_integer():
        raise TypeError(f"item_popularity column '{item_col}' must be integer typed.")

    if item_popularity[item_col].n_unique() != item_popularity.height:
        raise ValueError(f"item_popularity contains duplicate '{item_col}' values.")

    item_indices = item_popularity[item_col].cast(pl.Int64).to_numpy()

    if np.any(item_indices < 0):
        raise ValueError("item_popularity contains negative item indices.")

    max_item_index = int(item_indices.max())

    n_items = max_item_index + 1

    flattened = _validate_recommendation_indices(
        recommendations=recommendations,
        n_items=n_items,
    )

    popularity_lookup = np.zeros(
        n_items,
        dtype=np.float64,
    )

    popularity_lookup[item_indices] = (
        item_popularity[count_col]
        .to_numpy()
        .astype(
            np.float64,
            copy=False,
        )
    )

    if np.any(popularity_lookup < 0):
        raise ValueError("item popularity counts cannot be negative.")

    recommendation_counts = popularity_lookup[flattened]

    probabilities = recommendation_counts / total_interactions

    novelty = -np.log2(
        np.clip(
            probabilities,
            1e-12,
            None,
        )
    )

    return {
        "avg_item_interactions": float(recommendation_counts.mean()),
        "median_item_interactions": float(np.median(recommendation_counts)),
        "avg_novelty_bits": float(novelty.mean()),
    }
