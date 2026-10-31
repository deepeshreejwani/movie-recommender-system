from __future__ import annotations

import torch
from torch import nn


class MatrixFactorization(nn.Module):
    """
    Latent-factor collaborative-filtering model.

    Score(u, i) =
        <user_embedding, item_embedding>
        + user_bias
        + item_bias
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int = 64,
    ) -> None:
        super().__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim

        self.user_embedding = nn.Embedding(
            n_users,
            embedding_dim,
        )

        self.item_embedding = nn.Embedding(
            n_items,
            embedding_dim,
        )

        self.user_bias = nn.Embedding(
            n_users,
            1,
        )

        self.item_bias = nn.Embedding(
            n_items,
            1,
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(
            self.user_embedding.weight,
            mean=0.0,
            std=0.01,
        )

        nn.init.normal_(
            self.item_embedding.weight,
            mean=0.0,
            std=0.01,
        )

        nn.init.zeros_(self.user_bias.weight)

        nn.init.zeros_(self.item_bias.weight)

    def forward(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
    ) -> torch.Tensor:
        user_vectors = self.user_embedding(user_ids)

        item_vectors = self.item_embedding(item_ids)

        interaction = (user_vectors * item_vectors).sum(dim=1)

        user_bias = self.user_bias(user_ids).squeeze(-1)

        item_bias = self.item_bias(item_ids).squeeze(-1)

        return interaction + user_bias + item_bias

    @torch.no_grad()
    def score_all(
        self,
        user_ids: torch.Tensor,
    ) -> torch.Tensor:
        user_vectors = self.user_embedding(user_ids)

        scores = user_vectors @ self.item_embedding.weight.T

        scores = scores + self.user_bias(user_ids) + self.item_bias.weight.T

        return scores
