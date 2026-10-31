from __future__ import annotations

import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm

from recsys.data.sampling import (
    UniformNegativeSampler,
)


def _validate_training_inputs(
    user_ids: np.ndarray,
    item_ids: np.ndarray,
    negative_sampler: UniformNegativeSampler,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Validate and normalize encoded positive interactions
    used for model training.

    Training data must contain only known non-negative
    user and item indices. Cold holdout items such as
    item_idx = -1 must never enter model training.
    """

    if not isinstance(batch_size, int):
        raise TypeError("batch_size must be an integer.")

    if batch_size <= 0:
        raise ValueError("batch_size must be > 0.")

    user_ids = np.asarray(
        user_ids,
        dtype=np.int64,
    )

    item_ids = np.asarray(
        item_ids,
        dtype=np.int64,
    )

    if user_ids.ndim != 1:
        raise ValueError("user_ids must be a 1D array.")

    if item_ids.ndim != 1:
        raise ValueError("item_ids must be a 1D array.")

    if user_ids.size != item_ids.size:
        raise ValueError("user_ids and item_ids must have the same length.")

    if user_ids.size == 0:
        raise ValueError("Training interactions cannot be empty.")

    if np.any(user_ids < 0):
        raise ValueError("Training user IDs must be non-negative.")

    if np.any(item_ids < 0):
        raise ValueError(
            "Training item IDs must be non-negative. "
            "Cold holdout items cannot be used for training."
        )

    if np.any(item_ids >= negative_sampler.n_items):
        raise ValueError("Training item IDs exceed the negative sampler catalog size.")

    return (
        user_ids,
        item_ids,
    )


def train_pointwise_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    user_ids: np.ndarray,
    item_ids: np.ndarray,
    negative_sampler: UniformNegativeSampler,
    batch_size: int,
    device: torch.device,
    seed: int,
    desc: str = "Training",
) -> float:
    """
    Train one epoch using pointwise binary classification.

    For each observed positive interaction, one uniformly
    sampled unobserved item is used as a negative example.
    """

    (
        user_ids,
        item_ids,
    ) = _validate_training_inputs(
        user_ids=user_ids,
        item_ids=item_ids,
        negative_sampler=negative_sampler,
        batch_size=batch_size,
    )

    model.train()

    criterion = nn.BCEWithLogitsLoss()

    rng = np.random.default_rng(seed)

    permutation = rng.permutation(user_ids.size)

    total_loss = 0.0
    total_examples = 0

    for start in tqdm(
        range(
            0,
            user_ids.size,
            batch_size,
        ),
        desc=desc,
    ):
        batch_indices = permutation[start : start + batch_size]

        positive_users = user_ids[batch_indices]

        positive_items = item_ids[batch_indices]

        negative_items = negative_sampler.sample(positive_users)

        batch_users = np.concatenate(
            [
                positive_users,
                positive_users,
            ]
        )

        batch_items = np.concatenate(
            [
                positive_items,
                negative_items,
            ]
        )

        batch_labels = np.concatenate(
            [
                np.ones(
                    positive_users.size,
                    dtype=np.float32,
                ),
                np.zeros(
                    positive_users.size,
                    dtype=np.float32,
                ),
            ]
        )

        shuffle_indices = rng.permutation(batch_users.size)

        batch_users = batch_users[shuffle_indices]

        batch_items = batch_items[shuffle_indices]

        batch_labels = batch_labels[shuffle_indices]

        user_tensor = torch.as_tensor(
            batch_users,
            dtype=torch.long,
            device=device,
        )

        item_tensor = torch.as_tensor(
            batch_items,
            dtype=torch.long,
            device=device,
        )

        label_tensor = torch.as_tensor(
            batch_labels,
            dtype=torch.float32,
            device=device,
        )

        optimizer.zero_grad(set_to_none=True)

        logits = model(
            user_tensor,
            item_tensor,
        )

        if logits.ndim != 1:
            logits = logits.reshape(-1)

        if logits.size(0) != label_tensor.size(0):
            raise RuntimeError(
                "Model output size does not match the number of pointwise labels."
            )

        loss = criterion(
            logits,
            label_tensor,
        )

        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite pointwise training loss detected.")

        loss.backward()

        optimizer.step()

        batch_examples = batch_labels.size

        total_loss += loss.item() * batch_examples

        total_examples += batch_examples

    if total_examples == 0:
        raise RuntimeError("No pointwise training examples were processed.")

    return total_loss / total_examples


def train_bpr_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    user_ids: np.ndarray,
    item_ids: np.ndarray,
    negative_sampler: UniformNegativeSampler,
    batch_size: int,
    device: torch.device,
    seed: int,
    desc: str = "BPR training",
) -> float:
    """
    Train one epoch using Bayesian Personalized Ranking.

    For each observed user-positive pair, sample one
    unobserved item and optimize:

        softplus(
            negative_score
            - positive_score
        )

    so the observed item receives a higher score.
    """

    (
        user_ids,
        item_ids,
    ) = _validate_training_inputs(
        user_ids=user_ids,
        item_ids=item_ids,
        negative_sampler=negative_sampler,
        batch_size=batch_size,
    )

    model.train()

    rng = np.random.default_rng(seed)

    permutation = rng.permutation(user_ids.size)

    total_loss = 0.0
    total_examples = 0

    for start in tqdm(
        range(
            0,
            user_ids.size,
            batch_size,
        ),
        desc=desc,
    ):
        batch_indices = permutation[start : start + batch_size]

        batch_users = user_ids[batch_indices]

        positive_items = item_ids[batch_indices]

        negative_items = negative_sampler.sample(batch_users)

        user_tensor = torch.as_tensor(
            batch_users,
            dtype=torch.long,
            device=device,
        )

        positive_tensor = torch.as_tensor(
            positive_items,
            dtype=torch.long,
            device=device,
        )

        negative_tensor = torch.as_tensor(
            negative_items,
            dtype=torch.long,
            device=device,
        )

        optimizer.zero_grad(set_to_none=True)

        positive_scores = model(
            user_tensor,
            positive_tensor,
        )

        negative_scores = model(
            user_tensor,
            negative_tensor,
        )

        positive_scores = positive_scores.reshape(-1)

        negative_scores = negative_scores.reshape(-1)

        if (
            positive_scores.size(0) != batch_users.size
            or negative_scores.size(0) != batch_users.size
        ):
            raise RuntimeError(
                "Model output size does not match the BPR training batch size."
            )

        loss = torch.nn.functional.softplus(negative_scores - positive_scores).mean()

        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite BPR training loss detected.")

        loss.backward()

        optimizer.step()

        batch_examples = batch_users.size

        total_loss += loss.item() * batch_examples

        total_examples += batch_examples

    if total_examples == 0:
        raise RuntimeError("No BPR training examples were processed.")

    return total_loss / total_examples
