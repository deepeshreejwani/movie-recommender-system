from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import polars as pl
import torch
from tqdm.auto import tqdm

from recsys.evaluation.metrics import leave_one_out_metrics

_METRIC_NAMES = ("precision", "recall", "hitrate", "mrr", "ndcg")


@dataclass
class SeenItemsCSR:
    indptr: np.ndarray
    indices: np.ndarray


def _validate_required_columns(
    frame: pl.DataFrame, columns: tuple[str, ...], frame_name: str
) -> None:
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{frame_name} is missing required columns: {sorted(missing)}")
    for column in columns:
        null_count = frame[column].null_count()
        if null_count > 0:
            raise ValueError(
                f"{frame_name} contains {null_count} null values in '{column}'."
            )


def _validate_positive_int(value: int, name: str) -> None:
    if value < 1:
        raise ValueError(f"{name} must be >= 1.")


def _validate_non_negative_int(value: int, name: str) -> None:
    if value < 0:
        raise ValueError(f"{name} must be >= 0.")


def _validate_ks(ks: tuple[int, ...]) -> None:
    if not ks:
        raise ValueError("ks cannot be empty.")
    if any(k < 1 for k in ks):
        raise ValueError("Every evaluation K must be >= 1.")


def _validate_target_items(
    target_items: np.ndarray, n_users: int, n_items: int | None = None
) -> None:
    if target_items.ndim != 1:
        raise ValueError("target_items must be a one-dimensional array.")
    if target_items.shape[0] != n_users:
        raise ValueError("target_items length must equal n_users.")
    if np.any(target_items < -1):
        raise ValueError("Target item indices must be >= -1.")
    if n_items is not None:
        warm_targets = target_items[target_items >= 0]
        if warm_targets.size > 0 and np.any(warm_targets >= n_items):
            raise ValueError(
                "Holdout contains target item indices outside the model catalog."
            )


def _validate_seen_items_structure(seen_items: SeenItemsCSR, n_users: int) -> None:
    if seen_items.indptr.ndim != 1:
        raise ValueError("seen_items.indptr must be one-dimensional.")
    if seen_items.indices.ndim != 1:
        raise ValueError("seen_items.indices must be one-dimensional.")
    if seen_items.indptr.shape[0] != n_users + 1:
        raise ValueError("seen_items.indptr length must equal n_users + 1.")
    if seen_items.indptr[0] != 0:
        raise ValueError("seen_items.indptr must start at 0.")
    if np.any(np.diff(seen_items.indptr) < 0):
        raise ValueError("seen_items.indptr must be non-decreasing.")
    if int(seen_items.indptr[-1]) != seen_items.indices.shape[0]:
        raise ValueError(
            "seen_items.indptr[-1] must equal the number of seen item indices."
        )


def _validate_evaluation_inputs(
    target_items: np.ndarray,
    seen_items: SeenItemsCSR,
    n_users: int,
    batch_size: int,
    ks: tuple[int, ...],
) -> None:
    _validate_positive_int(n_users, "n_users")
    _validate_positive_int(batch_size, "batch_size")
    _validate_ks(ks)
    _validate_target_items(target_items=target_items, n_users=n_users)
    _validate_seen_items_structure(seen_items=seen_items, n_users=n_users)


def _validate_catalog(
    target_items: np.ndarray, seen_items: SeenItemsCSR, n_items: int
) -> None:
    _validate_positive_int(n_items, "n_items")
    _validate_target_items(
        target_items=target_items, n_users=target_items.shape[0], n_items=n_items
    )
    if seen_items.indices.size > 0:
        if np.any(seen_items.indices < 0):
            raise ValueError("Seen-item history cannot contain negative item indices.")
        if np.any(seen_items.indices >= n_items):
            raise ValueError(
                "Seen-item history contains item indices outside the model catalog."
            )


def _metric_sums(ks: tuple[int, ...]) -> dict[str, float]:
    return {f"{metric}@{k}": 0.0 for k in ks for metric in _METRIC_NAMES}


def _synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def build_seen_items_csr(
    interactions: pl.DataFrame,
    n_users: int,
    user_col: str = "user_idx",
    item_col: str = "item_idx",
) -> SeenItemsCSR:
    """
    Build CSR-style user history for efficient masking.

    Seen-item histories must contain only mapped non-negative item indices.
    Cold holdout targets encoded as -1 must never be included here.
    """
    _validate_positive_int(n_users, "n_users")
    if interactions.is_empty():
        raise ValueError("interactions cannot be empty.")
    _validate_required_columns(
        frame=interactions, columns=(user_col, item_col), frame_name="interactions"
    )
    sorted_interactions = (
        interactions.select(user_col, item_col).unique().sort(user_col, item_col)
    )
    user_ids = sorted_interactions[user_col].to_numpy().astype(np.int64, copy=False)
    item_ids = sorted_interactions[item_col].to_numpy().astype(np.int64, copy=False)
    if np.any((user_ids < 0) | (user_ids >= n_users)):
        raise ValueError("Seen interactions contain user indices outside [0, n_users).")
    if np.any(item_ids < 0):
        raise ValueError("Seen interactions cannot contain negative item indices.")
    counts = np.bincount(user_ids, minlength=n_users)
    indptr = np.zeros(n_users + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])
    return SeenItemsCSR(indptr=indptr, indices=item_ids)


def build_target_array(
    holdout: pl.DataFrame,
    n_users: int,
    user_col: str = "user_idx",
    item_col: str = "item_idx",
) -> np.ndarray:
    """
    Create exactly one evaluation target per user.

    `item_idx == -1` is a valid cold-item target and contributes a miss.
    A separate internal sentinel is used to detect genuinely missing targets.
    """
    _validate_positive_int(n_users, "n_users")
    if holdout.is_empty():
        raise ValueError("holdout cannot be empty.")
    _validate_required_columns(
        frame=holdout, columns=(user_col, item_col), frame_name="holdout"
    )
    if holdout.height != n_users:
        raise ValueError(
            "Leave-one-out evaluation requires exactly one target per user."
        )
    user_counts = holdout.group_by(user_col).len()
    if user_counts["len"].min() != 1 or user_counts["len"].max() != 1:
        raise ValueError(
            "Leave-one-out evaluation requires exactly one target per user."
        )
    missing_target = -2
    targets = np.full(n_users, missing_target, dtype=np.int64)
    users = holdout[user_col].to_numpy().astype(np.int64, copy=False)
    items = holdout[item_col].to_numpy().astype(np.int64, copy=False)
    if np.any((users < 0) | (users >= n_users)):
        raise ValueError("Holdout contains invalid user indices.")
    if np.any(items < -1):
        raise ValueError("Target item indices must be >= -1.")
    targets[users] = items
    if np.any(targets == missing_target):
        raise ValueError("At least one user is missing a holdout target.")
    return targets


def mask_seen_items(
    scores: torch.Tensor,
    user_ids: np.ndarray,
    seen_items: SeenItemsCSR,
) -> None:
    """Mask seen items in-place before Top-K selection."""

    if scores.ndim != 2:
        raise ValueError("scores must have shape [batch_size, n_items].")

    if user_ids.ndim != 1:
        raise ValueError("user_ids must be one-dimensional.")

    if scores.size(0) != len(user_ids):
        raise ValueError("scores and user_ids must contain the same number of users.")

    n_users = seen_items.indptr.shape[0] - 1

    if np.any((user_ids < 0) | (user_ids >= n_users)):
        raise ValueError("user_ids contain indices outside the seen-item history.")

    lengths = seen_items.indptr[user_ids + 1] - seen_items.indptr[user_ids]

    if lengths.sum() == 0:
        return

    row_indices = np.repeat(
        np.arange(
            len(user_ids),
            dtype=np.int64,
        ),
        lengths,
    )

    item_chunks = [
        seen_items.indices[seen_items.indptr[user_id] : seen_items.indptr[user_id + 1]]
        for user_id in user_ids
        if seen_items.indptr[user_id + 1] > seen_items.indptr[user_id]
    ]

    # np.load(..., mmap_mode="r") can return read-only arrays.
    # Materialize a writable copy before torch.as_tensor().
    item_indices = np.array(
        np.concatenate(item_chunks),
        dtype=np.int64,
        copy=True,
    )

    if np.any((item_indices < 0) | (item_indices >= scores.size(1))):
        raise ValueError(
            "Seen-item history contains indices outside the score catalog."
        )

    row_tensor = torch.as_tensor(
        row_indices,
        device=scores.device,
        dtype=torch.long,
    )

    item_tensor = torch.as_tensor(
        item_indices,
        device=scores.device,
        dtype=torch.long,
    )

    scores[
        row_tensor,
        item_tensor,
    ] = -torch.inf


def validate_finite_scores(
    scores: torch.Tensor,
    context: str,
) -> None:
    """Reject numerical corruption before deliberate seen-item masking."""

    if not torch.isfinite(scores).all():
        raise RuntimeError(f"{context} contain non-finite values before masking.")


def _topk_from_masked_scores(
    scores: torch.Tensor,
    k: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
]:
    """
    Select Top-K indices while preserving which positions are valid.

    Seen items are represented by ``-inf`` after ``mask_seen_items``.
    If a user has fewer than ``k`` unseen items, ``torch.topk`` still
    returns fixed-width indices for those ``-inf`` positions. The
    returned boolean mask distinguishes real candidates from those
    placeholders so they can never become hits after reranking.
    """

    if scores.ndim != 2:
        raise ValueError("scores must have shape [batch_size, candidate_count].")

    _validate_positive_int(
        k,
        "k",
    )

    if k > scores.size(1):
        raise ValueError("k cannot exceed the score width.")

    invalid_scores = torch.isnan(scores) | torch.isposinf(scores)

    if invalid_scores.any():
        raise ValueError("Scores contain NaN or positive infinity.")

    top_scores, top_indices = torch.topk(
        scores,
        k=k,
        dim=1,
    )

    valid_mask = ~torch.isneginf(top_scores)

    return (
        top_indices,
        valid_mask,
    )


def _pad_invalid_recommendations(
    recommendations: torch.Tensor,
    valid_mask: torch.Tensor,
    padding_index: int,
) -> torch.Tensor:
    """
    Replace invalid fixed-width Top-K positions with a catalog-external ID.

    Evaluation metrics require non-negative recommendation IDs. Using
    ``n_items`` as the padding value guarantees that a padded position
    cannot equal a warm target in ``[0, n_items)`` or the cold target
    sentinel ``-1``.
    """

    if recommendations.shape != valid_mask.shape:
        raise ValueError("recommendations and valid_mask must have the same shape.")

    if padding_index < 0:
        raise ValueError("padding_index must be non-negative.")

    return recommendations.masked_fill(
        ~valid_mask,
        padding_index,
    )


def _require_complete_topk(
    valid_mask: torch.Tensor,
    user_ids: np.ndarray,
    k: int,
    context: str,
) -> None:
    """
    Require every user to have ``k`` valid unseen recommendations.

    Fixed-width recommendation generation cannot safely represent
    fewer than ``k`` unseen items. Evaluation paths may instead pad
    invalid positions as guaranteed misses.
    """

    if valid_mask.ndim != 2:
        raise ValueError("valid_mask must have shape [batch_size, k].")

    if valid_mask.size(0) != len(user_ids):
        raise ValueError(
            "valid_mask and user_ids must contain the same number of users."
        )

    valid_counts = valid_mask.sum(dim=1)

    invalid_rows = torch.nonzero(
        valid_counts < k,
        as_tuple=False,
    ).flatten()

    if invalid_rows.numel() == 0:
        return

    row = int(invalid_rows[0].item())

    user_idx = int(user_ids[row])

    available = int(valid_counts[row].item())

    raise ValueError(
        f"{context} cannot return {k} unseen items for "
        f"user_idx={user_idx}; only {available} are available."
    )


@torch.no_grad()
def evaluate_model(
    model,
    target_items: np.ndarray,
    seen_items: SeenItemsCSR,
    n_users: int,
    batch_size: int,
    ks: tuple[int, ...],
    device: torch.device,
    desc: str = "Evaluating",
) -> dict[str, float]:
    """
    Evaluate a recommender using batched full-catalog ranking.

    When a dense-history user has fewer than ``max(ks)`` unseen
    items, fixed-width Top-K positions backed by masked ``-inf``
    scores are replaced with the catalog-external ID ``n_items``.
    Those positions are therefore guaranteed misses.
    """

    _validate_evaluation_inputs(
        target_items=target_items,
        seen_items=seen_items,
        n_users=n_users,
        batch_size=batch_size,
        ks=ks,
    )

    model.eval()

    max_k = max(ks)

    metric_sums = _metric_sums(ks)

    total_users = 0

    n_items: int | None = None

    for start in tqdm(
        range(
            0,
            n_users,
            batch_size,
        ),
        desc=desc,
    ):
        end = min(
            start + batch_size,
            n_users,
        )

        batch_users_np = np.arange(
            start,
            end,
            dtype=np.int64,
        )

        batch_users = torch.as_tensor(
            batch_users_np,
            device=device,
            dtype=torch.long,
        )

        scores = model.score_all(batch_users)

        if scores.ndim != 2 or scores.size(0) != end - start:
            raise ValueError("model.score_all must return [batch_size, n_items].")

        observed_n_items = scores.size(1)

        if n_items is None:
            n_items = observed_n_items

            if max_k > n_items:
                raise ValueError("Maximum evaluation K cannot exceed catalog size.")

            _validate_catalog(
                target_items=target_items,
                seen_items=seen_items,
                n_items=n_items,
            )

        elif observed_n_items != n_items:
            raise ValueError(
                "model.score_all returned inconsistent catalog sizes "
                "across evaluation batches."
            )

        validate_finite_scores(
            scores=scores,
            context="Model scores",
        )

        mask_seen_items(
            scores=scores,
            user_ids=batch_users_np,
            seen_items=seen_items,
        )

        (
            recommendations,
            valid_recommendations,
        ) = _topk_from_masked_scores(
            scores=scores,
            k=max_k,
        )

        recommendations = _pad_invalid_recommendations(
            recommendations=recommendations,
            valid_mask=valid_recommendations,
            padding_index=n_items,
        )

        batch_targets = torch.as_tensor(
            target_items[start:end],
            device=device,
            dtype=torch.long,
        )

        batch_metrics = leave_one_out_metrics(
            recommendations=recommendations,
            target_items=batch_targets,
            ks=ks,
        )

        for (
            metric_name,
            value,
        ) in batch_metrics.items():
            metric_sums[metric_name] += value

        total_users += end - start

        del scores
        del recommendations

    return {
        metric: (value / total_users)
        for (
            metric,
            value,
        ) in metric_sums.items()
    }


@torch.no_grad()
def benchmark_full_catalog_scoring(
    model,
    batch_size: int,
    n_items: int,
    device: torch.device,
    warmup_runs: int = 2,
    benchmark_runs: int = 5,
) -> dict[str, float]:
    """Benchmark full-catalog scoring throughput for a user batch."""
    _validate_positive_int(batch_size, "batch_size")
    _validate_positive_int(n_items, "n_items")
    _validate_non_negative_int(warmup_runs, "warmup_runs")
    _validate_positive_int(benchmark_runs, "benchmark_runs")
    model.eval()
    user_ids = torch.arange(batch_size, device=device, dtype=torch.long)
    observed_n_items: int | None = None
    for _ in range(warmup_runs):
        scores = model.score_all(user_ids)
        if scores.ndim != 2 or scores.size(0) != batch_size:
            raise ValueError("model.score_all must return [batch_size, n_items].")
        observed_n_items = scores.size(1)
        del scores
    if observed_n_items is None:
        scores = model.score_all(user_ids)
        if scores.ndim != 2 or scores.size(0) != batch_size:
            raise ValueError("model.score_all must return [batch_size, n_items].")
        observed_n_items = scores.size(1)
        del scores
    if observed_n_items != n_items:
        raise ValueError(
            f"Configured n_items={n_items} does not match "
            f"model output catalog size={observed_n_items}."
        )
    _synchronize(device)
    start = time.perf_counter()
    for _ in range(benchmark_runs):
        scores = model.score_all(user_ids)
        del scores
    _synchronize(device)
    elapsed = time.perf_counter() - start
    avg_seconds = elapsed / benchmark_runs
    users_per_second = batch_size / avg_seconds
    user_item_scores_per_second = batch_size * n_items / avg_seconds
    return {
        "batch_size": batch_size,
        "items": n_items,
        "avg_batch_seconds": avg_seconds,
        "users_per_second": users_per_second,
        "million_scores_per_second": user_item_scores_per_second / 1000000,
    }


@torch.no_grad()
def evaluate_embedding_retrieval(
    model,
    item_embeddings: torch.Tensor,
    target_items: np.ndarray,
    seen_items: SeenItemsCSR,
    n_users: int,
    batch_size: int,
    ks: tuple[int, ...],
    device: torch.device,
    desc: str = "Retrieval evaluation",
) -> dict[str, float]:
    """
    Evaluate embedding retrieval using precomputed item embeddings.

    Dense-history users with fewer than ``max(ks)`` unseen items
    receive catalog-external padding IDs for invalid Top-K slots,
    guaranteeing that masked seen items remain misses.
    """

    _validate_evaluation_inputs(
        target_items=target_items,
        seen_items=seen_items,
        n_users=n_users,
        batch_size=batch_size,
        ks=ks,
    )

    if item_embeddings.ndim != 2:
        raise ValueError("item_embeddings must have shape [n_items, embedding_dim].")

    n_items = item_embeddings.size(0)

    _validate_positive_int(
        n_items,
        "n_items",
    )

    max_k = max(ks)

    if max_k > n_items:
        raise ValueError("Maximum evaluation K cannot exceed catalog size.")

    _validate_catalog(
        target_items=target_items,
        seen_items=seen_items,
        n_items=n_items,
    )

    model.eval()

    item_embeddings = item_embeddings.to(
        device=device,
        dtype=torch.float32,
    )

    metric_sums = _metric_sums(ks)

    total_users = 0

    for start in tqdm(
        range(
            0,
            n_users,
            batch_size,
        ),
        desc=desc,
    ):
        end = min(
            start + batch_size,
            n_users,
        )

        batch_users_np = np.arange(
            start,
            end,
            dtype=np.int64,
        )

        batch_users = torch.as_tensor(
            batch_users_np,
            dtype=torch.long,
            device=device,
        )

        user_embeddings = model.encode_users(batch_users)

        scores = user_embeddings @ item_embeddings.T

        validate_finite_scores(
            scores=scores,
            context="Model scores",
        )

        mask_seen_items(
            scores=scores,
            user_ids=batch_users_np,
            seen_items=seen_items,
        )

        (
            recommendations,
            valid_recommendations,
        ) = _topk_from_masked_scores(
            scores=scores,
            k=max_k,
        )

        recommendations = _pad_invalid_recommendations(
            recommendations=recommendations,
            valid_mask=valid_recommendations,
            padding_index=n_items,
        )

        batch_targets = torch.as_tensor(
            target_items[start:end],
            dtype=torch.long,
            device=device,
        )

        batch_metrics = leave_one_out_metrics(
            recommendations=recommendations,
            target_items=batch_targets,
            ks=ks,
        )

        for (
            metric_name,
            value,
        ) in batch_metrics.items():
            metric_sums[metric_name] += value

        total_users += end - start

        del user_embeddings
        del scores
        del recommendations

    return {
        metric: (value / total_users)
        for (
            metric,
            value,
        ) in metric_sums.items()
    }


@torch.no_grad()
def benchmark_embedding_retrieval(
    model,
    item_embeddings: torch.Tensor,
    batch_size: int,
    device: torch.device,
    warmup_runs: int = 2,
    benchmark_runs: int = 5,
) -> dict[str, float]:
    """Benchmark exact embedding retrieval with precomputed item embeddings."""
    _validate_positive_int(batch_size, "batch_size")
    _validate_non_negative_int(warmup_runs, "warmup_runs")
    _validate_positive_int(benchmark_runs, "benchmark_runs")
    if item_embeddings.ndim != 2 or item_embeddings.size(0) < 1:
        raise ValueError(
            "item_embeddings must have shape "
            "[n_items, embedding_dim] with at least one item."
        )
    model.eval()
    item_embeddings = item_embeddings.to(device=device, dtype=torch.float32)
    user_ids = torch.arange(batch_size, dtype=torch.long, device=device)
    for _ in range(warmup_runs):
        user_embeddings = model.encode_users(user_ids)
        scores = user_embeddings @ item_embeddings.T
        del user_embeddings
        del scores
    _synchronize(device)
    start_time = time.perf_counter()
    for _ in range(benchmark_runs):
        user_embeddings = model.encode_users(user_ids)
        scores = user_embeddings @ item_embeddings.T
        del user_embeddings
        del scores
    _synchronize(device)
    elapsed = time.perf_counter() - start_time
    avg_batch_seconds = elapsed / benchmark_runs
    n_items = item_embeddings.size(0)
    users_per_second = batch_size / avg_batch_seconds
    scores_per_second = batch_size * n_items / avg_batch_seconds
    return {
        "batch_size": batch_size,
        "items": n_items,
        "avg_batch_seconds": avg_batch_seconds,
        "users_per_second": users_per_second,
        "million_scores_per_second": scores_per_second / 1000000,
    }


@torch.no_grad()
def score_candidate_pairs(
    model, user_ids: torch.Tensor, candidate_items: torch.Tensor, score_batch_size: int
) -> torch.Tensor:
    """
    Score user-candidate pairs in chunks.

    `user_ids` has shape [batch_size].
    `candidate_items` has shape [batch_size, candidate_size].
    """
    _validate_positive_int(score_batch_size, "score_batch_size")
    if user_ids.ndim != 1:
        raise ValueError("user_ids must have shape [batch_size].")
    if candidate_items.ndim != 2:
        raise ValueError(
            "candidate_items must have shape [batch_size, candidate_size]."
        )
    batch_size, candidate_size = candidate_items.shape
    if batch_size != user_ids.size(0):
        raise ValueError(
            "user_ids and candidate_items must contain the same batch size."
        )
    if batch_size < 1:
        raise ValueError("Candidate scoring batch cannot be empty.")
    if candidate_size < 1:
        raise ValueError("candidate_size must be >= 1.")
    expanded_users = user_ids.unsqueeze(1).expand(-1, candidate_size).reshape(-1)
    flattened_items = candidate_items.reshape(-1)
    score_chunks: list[torch.Tensor] = []
    for start in range(0, flattened_items.size(0), score_batch_size):
        end = min(start + score_batch_size, flattened_items.size(0))
        chunk_scores = model(expanded_users[start:end], flattened_items[start:end])
        score_chunks.append(chunk_scores)
    scores = torch.cat(score_chunks, dim=0).reshape(batch_size, candidate_size)

    if not torch.isfinite(scores).all():
        raise RuntimeError("Ranker scores contain non-finite values.")

    return scores


@torch.no_grad()
def evaluate_retrieval_reranking(
    retrieval_model,
    ranker,
    item_embeddings: torch.Tensor,
    target_items: np.ndarray,
    seen_items: SeenItemsCSR,
    n_users: int,
    user_batch_size: int,
    candidate_size: int,
    ks: tuple[int, ...],
    score_batch_size: int,
    device: torch.device,
    desc: str = "Retrieval + ranking",
) -> dict[str, object]:
    """
    Evaluate an end-to-end retrieval and reranking pipeline.

    Candidate validity produced after seen-item masking is preserved
    through the ranker. Masked candidates are assigned ``-inf`` ranker
    scores and can never become candidate hits or final metric hits.
    """

    _validate_evaluation_inputs(
        target_items=target_items,
        seen_items=seen_items,
        n_users=n_users,
        batch_size=user_batch_size,
        ks=ks,
    )

    _validate_positive_int(
        candidate_size,
        "candidate_size",
    )

    _validate_positive_int(
        score_batch_size,
        "score_batch_size",
    )

    if item_embeddings.ndim != 2:
        raise ValueError("item_embeddings must have shape [n_items, embedding_dim].")

    n_items = item_embeddings.size(0)

    if candidate_size > n_items:
        raise ValueError("candidate_size cannot exceed catalog size.")

    max_k = max(ks)

    if max_k > candidate_size:
        raise ValueError("Maximum evaluation K cannot exceed candidate_size.")

    _validate_catalog(
        target_items=target_items,
        seen_items=seen_items,
        n_items=n_items,
    )

    retrieval_model.eval()
    ranker.eval()

    item_embeddings = item_embeddings.to(
        device=device,
        dtype=torch.float32,
    )

    metric_sums = _metric_sums(ks)

    total_users = 0
    candidate_hits = 0

    start_time = time.perf_counter()

    for start in tqdm(
        range(
            0,
            n_users,
            user_batch_size,
        ),
        desc=desc,
    ):
        end = min(
            start + user_batch_size,
            n_users,
        )

        batch_users_np = np.arange(
            start,
            end,
            dtype=np.int64,
        )

        batch_users = torch.as_tensor(
            batch_users_np,
            dtype=torch.long,
            device=device,
        )

        user_embeddings = retrieval_model.encode_users(batch_users)

        retrieval_scores = user_embeddings @ item_embeddings.T

        validate_finite_scores(
            scores=retrieval_scores,
            context="Raw retrieval scores",
        )

        mask_seen_items(
            scores=retrieval_scores,
            user_ids=batch_users_np,
            seen_items=seen_items,
        )

        (
            candidate_items,
            candidate_valid,
        ) = _topk_from_masked_scores(
            scores=retrieval_scores,
            k=candidate_size,
        )

        del user_embeddings
        del retrieval_scores

        batch_targets = torch.as_tensor(
            target_items[start:end],
            dtype=torch.long,
            device=device,
        )

        candidate_hits += int(
            (candidate_items.eq(batch_targets.unsqueeze(1)) & candidate_valid)
            .any(dim=1)
            .sum()
            .item()
        )

        candidate_scores = score_candidate_pairs(
            model=ranker,
            user_ids=batch_users,
            candidate_items=candidate_items,
            score_batch_size=score_batch_size,
        )

        candidate_scores.masked_fill_(
            ~candidate_valid,
            -torch.inf,
        )

        (
            reranked_positions,
            reranked_valid,
        ) = _topk_from_masked_scores(
            scores=candidate_scores,
            k=max_k,
        )

        recommendations = candidate_items.gather(
            dim=1,
            index=reranked_positions,
        )

        recommendation_valid = (
            candidate_valid.gather(
                dim=1,
                index=reranked_positions,
            )
            & reranked_valid
        )

        recommendations = _pad_invalid_recommendations(
            recommendations=recommendations,
            valid_mask=recommendation_valid,
            padding_index=n_items,
        )

        batch_metrics = leave_one_out_metrics(
            recommendations=recommendations,
            target_items=batch_targets,
            ks=ks,
        )

        for (
            metric_name,
            value,
        ) in batch_metrics.items():
            metric_sums[metric_name] += value

        total_users += end - start

        del candidate_items
        del candidate_valid
        del candidate_scores
        del recommendations

    _synchronize(device)

    elapsed_seconds = time.perf_counter() - start_time

    metrics = {
        metric: (value / total_users)
        for (
            metric,
            value,
        ) in metric_sums.items()
    }

    return {
        "candidate_size": (candidate_size),
        "candidate_recall": (candidate_hits / total_users),
        "metrics": metrics,
        "elapsed_seconds": (elapsed_seconds),
        "users_per_second": (total_users / elapsed_seconds),
    }


@torch.no_grad()
def evaluate_reranking_candidate_sizes(
    retrieval_model,
    ranker,
    item_embeddings: torch.Tensor,
    target_items: np.ndarray,
    seen_items: SeenItemsCSR,
    n_users: int,
    user_batch_size: int,
    candidate_sizes: tuple[int, ...],
    ks: tuple[int, ...],
    score_batch_size: int,
    device: torch.device,
    desc: str = "Candidate-size evaluation",
) -> dict[
    int | str,
    dict[str, object],
]:
    """
    Evaluate multiple retrieval candidate sizes efficiently.

    The largest pool is retrieved and scored once per user batch.
    Smaller pools reuse prefixes of the same retrieval ranking.
    Candidate validity is preserved through every reranking prefix.
    """

    _validate_evaluation_inputs(
        target_items=target_items,
        seen_items=seen_items,
        n_users=n_users,
        batch_size=user_batch_size,
        ks=ks,
    )

    _validate_positive_int(
        score_batch_size,
        "score_batch_size",
    )

    if not candidate_sizes:
        raise ValueError("candidate_sizes cannot be empty.")

    if any(candidate_size < 1 for candidate_size in candidate_sizes):
        raise ValueError("Every candidate size must be >= 1.")

    candidate_sizes = tuple(sorted(set(candidate_sizes)))

    if item_embeddings.ndim != 2:
        raise ValueError("item_embeddings must have shape [n_items, embedding_dim].")

    n_items = item_embeddings.size(0)

    max_candidate_size = max(candidate_sizes)

    if max_candidate_size > n_items:
        raise ValueError("Candidate size cannot exceed catalog size.")

    max_k = max(ks)

    if max_k > min(candidate_sizes):
        raise ValueError("Every candidate size must be >= max evaluation K.")

    _validate_catalog(
        target_items=target_items,
        seen_items=seen_items,
        n_items=n_items,
    )

    retrieval_model.eval()
    ranker.eval()

    item_embeddings = item_embeddings.to(
        device=device,
        dtype=torch.float32,
    )

    metric_sums = {
        candidate_size: _metric_sums(ks) for candidate_size in candidate_sizes
    }

    candidate_hits = {candidate_size: 0 for candidate_size in candidate_sizes}

    total_users = 0

    start_time = time.perf_counter()

    for start in tqdm(
        range(
            0,
            n_users,
            user_batch_size,
        ),
        desc=desc,
    ):
        end = min(
            start + user_batch_size,
            n_users,
        )

        batch_users_np = np.arange(
            start,
            end,
            dtype=np.int64,
        )

        batch_users = torch.as_tensor(
            batch_users_np,
            dtype=torch.long,
            device=device,
        )

        batch_targets = torch.as_tensor(
            target_items[start:end],
            dtype=torch.long,
            device=device,
        )

        user_embeddings = retrieval_model.encode_users(batch_users)

        retrieval_scores = user_embeddings @ item_embeddings.T

        validate_finite_scores(
            scores=retrieval_scores,
            context="Raw retrieval scores",
        )

        mask_seen_items(
            scores=retrieval_scores,
            user_ids=batch_users_np,
            seen_items=seen_items,
        )

        (
            max_candidates,
            max_candidate_valid,
        ) = _topk_from_masked_scores(
            scores=retrieval_scores,
            k=max_candidate_size,
        )

        del user_embeddings
        del retrieval_scores

        max_candidate_scores = score_candidate_pairs(
            model=ranker,
            user_ids=batch_users,
            candidate_items=max_candidates,
            score_batch_size=score_batch_size,
        )

        max_candidate_scores.masked_fill_(
            ~max_candidate_valid,
            -torch.inf,
        )

        for candidate_size in candidate_sizes:
            candidate_items = max_candidates[
                :,
                :candidate_size,
            ]

            candidate_valid = max_candidate_valid[
                :,
                :candidate_size,
            ]

            candidate_scores = max_candidate_scores[
                :,
                :candidate_size,
            ]

            candidate_hits[candidate_size] += int(
                (candidate_items.eq(batch_targets.unsqueeze(1)) & candidate_valid)
                .any(dim=1)
                .sum()
                .item()
            )

            (
                reranked_positions,
                reranked_valid,
            ) = _topk_from_masked_scores(
                scores=candidate_scores,
                k=max_k,
            )

            recommendations = candidate_items.gather(
                dim=1,
                index=reranked_positions,
            )

            recommendation_valid = (
                candidate_valid.gather(
                    dim=1,
                    index=reranked_positions,
                )
                & reranked_valid
            )

            recommendations = _pad_invalid_recommendations(
                recommendations=recommendations,
                valid_mask=recommendation_valid,
                padding_index=n_items,
            )

            batch_metrics = leave_one_out_metrics(
                recommendations=recommendations,
                target_items=batch_targets,
                ks=ks,
            )

            for (
                metric_name,
                value,
            ) in batch_metrics.items():
                metric_sums[candidate_size][metric_name] += value

        total_users += end - start

        del max_candidates
        del max_candidate_valid
        del max_candidate_scores

    _synchronize(device)

    elapsed_seconds = time.perf_counter() - start_time

    result: dict[
        int | str,
        dict[str, object],
    ] = {
        candidate_size: {
            "candidate_recall": (candidate_hits[candidate_size] / total_users),
            "metrics": {
                metric: (value / total_users)
                for (
                    metric,
                    value,
                ) in metric_sums[candidate_size].items()
            },
        }
        for candidate_size in candidate_sizes
    }

    result["_benchmark"] = {
        "elapsed_seconds": (elapsed_seconds),
        "users_per_second": (total_users / elapsed_seconds),
    }

    return result


@torch.no_grad()
def benchmark_candidate_ranking(
    ranker,
    batch_size: int,
    candidate_size: int,
    n_users: int,
    n_items: int,
    score_batch_size: int,
    device: torch.device,
    seed: int = 42,
    warmup_runs: int = 2,
    benchmark_runs: int = 5,
) -> dict[str, float]:
    """
    Benchmark neural candidate scoring throughput.

    Retrieval and Top-K candidate generation are excluded.
    """
    _validate_positive_int(batch_size, "batch_size")
    _validate_positive_int(candidate_size, "candidate_size")
    _validate_positive_int(n_users, "n_users")
    _validate_positive_int(n_items, "n_items")
    _validate_positive_int(score_batch_size, "score_batch_size")
    _validate_non_negative_int(warmup_runs, "warmup_runs")
    _validate_positive_int(benchmark_runs, "benchmark_runs")
    ranker.eval()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    user_ids = torch.randint(
        low=0, high=n_users, size=(batch_size,), generator=generator, dtype=torch.long
    ).to(device)
    candidate_items = torch.randint(
        low=0,
        high=n_items,
        size=(batch_size, candidate_size),
        generator=generator,
        dtype=torch.long,
    ).to(device)
    for _ in range(warmup_runs):
        scores = score_candidate_pairs(
            model=ranker,
            user_ids=user_ids,
            candidate_items=candidate_items,
            score_batch_size=score_batch_size,
        )
        del scores
    _synchronize(device)
    start_time = time.perf_counter()
    for _ in range(benchmark_runs):
        scores = score_candidate_pairs(
            model=ranker,
            user_ids=user_ids,
            candidate_items=candidate_items,
            score_batch_size=score_batch_size,
        )
        del scores
    _synchronize(device)
    elapsed_seconds = time.perf_counter() - start_time
    avg_batch_seconds = elapsed_seconds / benchmark_runs
    total_scores = batch_size * candidate_size
    return {
        "batch_size": batch_size,
        "candidate_size": candidate_size,
        "avg_batch_seconds": avg_batch_seconds,
        "batch_latency_ms": avg_batch_seconds * 1000,
        "users_per_second": batch_size / avg_batch_seconds,
        "million_scores_per_second": total_scores / avg_batch_seconds / 1000000,
    }


@torch.no_grad()
def generate_recommendations(
    retrieval_model,
    item_embeddings: torch.Tensor,
    seen_items: SeenItemsCSR,
    n_users: int,
    user_batch_size: int,
    top_k: int,
    device: torch.device,
    ranker=None,
    candidate_size: int | None = None,
    score_batch_size: int | None = None,
    desc: str = "Generating recommendations",
) -> np.ndarray:
    """
    Generate retrieval-only or retrieval + reranking recommendations.

    Unlike metric evaluation, this function returns real item IDs and
    therefore cannot pad a dense user's result with a synthetic catalog
    ID. It raises when a user has fewer than ``top_k`` unseen items.
    """

    _validate_positive_int(
        n_users,
        "n_users",
    )

    _validate_positive_int(
        user_batch_size,
        "user_batch_size",
    )

    _validate_positive_int(
        top_k,
        "top_k",
    )

    _validate_seen_items_structure(
        seen_items=seen_items,
        n_users=n_users,
    )

    if item_embeddings.ndim != 2:
        raise ValueError("item_embeddings must have shape [n_items, embedding_dim].")

    n_items = item_embeddings.size(0)

    _validate_positive_int(
        n_items,
        "n_items",
    )

    if top_k > n_items:
        raise ValueError("top_k cannot exceed catalog size.")

    if seen_items.indices.size > 0:
        if np.any(seen_items.indices < 0):
            raise ValueError("Seen-item history cannot contain negative item indices.")

        if np.any(seen_items.indices >= n_items):
            raise ValueError(
                "Seen-item history contains item indices outside the model catalog."
            )

    retrieval_model.eval()

    if ranker is not None:
        ranker.eval()

        if candidate_size is None:
            raise ValueError("candidate_size is required when a ranker is provided.")

        if score_batch_size is None:
            raise ValueError("score_batch_size is required when a ranker is provided.")

        _validate_positive_int(
            candidate_size,
            "candidate_size",
        )

        _validate_positive_int(
            score_batch_size,
            "score_batch_size",
        )

        if candidate_size > n_items:
            raise ValueError("candidate_size cannot exceed catalog size.")

        if top_k > candidate_size:
            raise ValueError("top_k cannot exceed candidate_size.")

    item_embeddings = item_embeddings.to(
        device=device,
        dtype=torch.float32,
    )

    recommendation_batches: list[np.ndarray] = []

    for start in tqdm(
        range(
            0,
            n_users,
            user_batch_size,
        ),
        desc=desc,
    ):
        end = min(
            start + user_batch_size,
            n_users,
        )

        batch_users_np = np.arange(
            start,
            end,
            dtype=np.int64,
        )

        batch_users = torch.as_tensor(
            batch_users_np,
            dtype=torch.long,
            device=device,
        )

        user_embeddings = retrieval_model.encode_users(batch_users)

        retrieval_scores = user_embeddings @ item_embeddings.T

        validate_finite_scores(
            scores=retrieval_scores,
            context="Raw retrieval scores",
        )

        mask_seen_items(
            scores=retrieval_scores,
            user_ids=batch_users_np,
            seen_items=seen_items,
        )

        if ranker is None:
            (
                recommendations,
                recommendation_valid,
            ) = _topk_from_masked_scores(
                scores=retrieval_scores,
                k=top_k,
            )

            _require_complete_topk(
                valid_mask=recommendation_valid,
                user_ids=batch_users_np,
                k=top_k,
                context="Recommendation generation",
            )

            del retrieval_scores

        else:
            (
                candidate_items,
                candidate_valid,
            ) = _topk_from_masked_scores(
                scores=retrieval_scores,
                k=candidate_size,
            )

            del retrieval_scores

            candidate_scores = score_candidate_pairs(
                model=ranker,
                user_ids=batch_users,
                candidate_items=candidate_items,
                score_batch_size=score_batch_size,
            )

            candidate_scores.masked_fill_(
                ~candidate_valid,
                -torch.inf,
            )

            (
                reranked_positions,
                reranked_valid,
            ) = _topk_from_masked_scores(
                scores=candidate_scores,
                k=top_k,
            )

            recommendations = candidate_items.gather(
                dim=1,
                index=reranked_positions,
            )

            recommendation_valid = (
                candidate_valid.gather(
                    dim=1,
                    index=reranked_positions,
                )
                & reranked_valid
            )

            _require_complete_topk(
                valid_mask=recommendation_valid,
                user_ids=batch_users_np,
                k=top_k,
                context="Recommendation generation",
            )

            del candidate_items
            del candidate_valid
            del candidate_scores

        recommendation_batches.append(
            recommendations.detach()
            .cpu()
            .numpy()
            .astype(
                np.int32,
                copy=False,
            )
        )

        del user_embeddings
        del recommendations

    return np.concatenate(
        recommendation_batches,
        axis=0,
    )
