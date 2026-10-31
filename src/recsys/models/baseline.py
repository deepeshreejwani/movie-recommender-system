from __future__ import annotations

import polars as pl
import torch


class PopularityRecommender:
    """Recommend globally popular training items."""

    def __init__(
        self,
        n_items: int,
        device: torch.device,
    ) -> None:
        if n_items <= 0:
            raise ValueError("n_items must be > 0.")

        self.n_items = n_items
        self.device = device

        self.item_scores: torch.Tensor | None = None

    def fit(
        self,
        train: pl.DataFrame,
        item_col: str = "item_idx",
    ) -> PopularityRecommender:
        """Fit item popularity scores from training interactions."""

        if train.is_empty():
            raise ValueError("train cannot be empty.")

        if item_col not in train.columns:
            raise ValueError(f"Missing item column: {item_col}")

        item_values = train[item_col].to_numpy()

        if item_values.min() < 0 or item_values.max() >= self.n_items:
            raise ValueError(f"Training item indices must be in [0, {self.n_items}).")

        popularity = train.group_by(item_col).len().sort(item_col)

        scores = torch.zeros(
            self.n_items,
            dtype=torch.float32,
            device=self.device,
        )

        item_indices = torch.as_tensor(
            popularity[item_col].to_numpy(),
            dtype=torch.long,
            device=self.device,
        )

        interaction_counts = torch.as_tensor(
            popularity["len"].to_numpy(),
            dtype=torch.float32,
            device=self.device,
        )

        scores[item_indices] = interaction_counts

        self.item_scores = scores

        return self

    def eval(
        self,
    ) -> PopularityRecommender:
        """
        No-op evaluation mode.

        Provided for compatibility with the common
        recommender evaluation interface.
        """

        return self

    @torch.no_grad()
    def score_all(
        self,
        user_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Return popularity scores for every item."""

        if self.item_scores is None:
            raise RuntimeError(
                "PopularityRecommender must be fitted before evaluation."
            )

        if user_ids.ndim != 1:
            raise ValueError("user_ids must be a 1D tensor.")

        if user_ids.numel() == 0:
            return torch.empty(
                (
                    0,
                    self.n_items,
                ),
                dtype=torch.float32,
                device=self.device,
            )

        return (
            self.item_scores.unsqueeze(0)
            .expand(
                user_ids.size(0),
                -1,
            )
            .clone()
        )


class RandomRecommender:
    """
    Random-ranking baseline.

    A dedicated random generator is used so the baseline does
    not depend on PyTorch's global random state.
    """

    def __init__(
        self,
        n_items: int,
        device: torch.device,
        seed: int = 42,
    ) -> None:
        if n_items <= 0:
            raise ValueError("n_items must be > 0.")

        self.n_items = n_items
        self.device = device
        self.seed = seed

        self._generator = torch.Generator(device="cpu")

        self._generator.manual_seed(self.seed)

    def fit(
        self,
        train: pl.DataFrame,
    ) -> RandomRecommender:
        """
        Reset the random baseline to its configured seed.

        The training data is intentionally unused because this
        baseline does not learn from interactions.
        """

        if train.is_empty():
            raise ValueError("train cannot be empty.")

        self._generator.manual_seed(self.seed)

        return self

    def eval(
        self,
    ) -> RandomRecommender:
        """
        No-op evaluation mode.

        Provided for compatibility with the common
        recommender evaluation interface.
        """

        return self

    @torch.no_grad()
    def score_all(
        self,
        user_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Generate random item scores for each supplied user."""

        if user_ids.ndim != 1:
            raise ValueError("user_ids must be a 1D tensor.")

        batch_size = user_ids.size(0)

        if batch_size == 0:
            return torch.empty(
                (
                    0,
                    self.n_items,
                ),
                dtype=torch.float32,
                device=self.device,
            )

        # Generate on CPU using our dedicated RNG.
        #
        # This avoids mutating PyTorch's global RNG and remains
        # compatible with CPU, CUDA, and MPS evaluation devices.
        scores = torch.rand(
            (
                batch_size,
                self.n_items,
            ),
            generator=self._generator,
            dtype=torch.float32,
            device="cpu",
        )

        return scores.to(self.device)
