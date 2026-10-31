from __future__ import annotations

import torch
from torch import nn


class TwoTowerRecommender(nn.Module):
    """
    Two-Tower retrieval model.

    User and item representations are produced independently and scored
    using an inner product, enabling efficient embedding retrieval.
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.user_embedding = nn.Embedding(
            n_users,
            embedding_dim,
        )

        self.item_embedding = nn.Embedding(
            n_items,
            embedding_dim,
        )

        self.user_tower = nn.Sequential(
            nn.Linear(
                embedding_dim,
                hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(
                hidden_dim,
                embedding_dim,
            ),
        )

        self.item_tower = nn.Sequential(
            nn.Linear(
                embedding_dim,
                hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(
                hidden_dim,
                embedding_dim,
            ),
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

        for tower in [
            self.user_tower,
            self.item_tower,
        ]:
            for module in tower:
                if isinstance(
                    module,
                    nn.Linear,
                ):
                    nn.init.xavier_uniform_(module.weight)

                    nn.init.zeros_(module.bias)

    def encode_users(
        self,
        user_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self.user_tower(self.user_embedding(user_ids))

    def encode_items(
        self,
        item_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self.item_tower(self.item_embedding(item_ids))

    def forward(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
    ) -> torch.Tensor:
        user_vectors = self.encode_users(user_ids)

        item_vectors = self.encode_items(item_ids)

        return (user_vectors * item_vectors).sum(dim=1)

    @torch.no_grad()
    def score_all(
        self,
        user_ids: torch.Tensor,
    ) -> torch.Tensor:
        user_vectors = self.encode_users(user_ids)

        item_ids = torch.arange(
            self.n_items,
            device=user_ids.device,
        )

        item_vectors = self.encode_items(item_ids)

        return user_vectors @ item_vectors.T

    @torch.no_grad()
    def get_user_embeddings(
        self,
        device: torch.device,
        batch_size: int = 4096,
    ) -> torch.Tensor:
        embeddings = []

        for start in range(
            0,
            self.n_users,
            batch_size,
        ):
            end = min(
                start + batch_size,
                self.n_users,
            )

            user_ids = torch.arange(
                start,
                end,
                device=device,
            )

            embeddings.append(self.encode_users(user_ids).cpu())

        return torch.cat(
            embeddings,
            dim=0,
        )

    @torch.no_grad()
    def get_item_embeddings(
        self,
        device: torch.device,
        batch_size: int = 4096,
    ) -> torch.Tensor:
        embeddings = []

        for start in range(
            0,
            self.n_items,
            batch_size,
        ):
            end = min(
                start + batch_size,
                self.n_items,
            )

            item_ids = torch.arange(
                start,
                end,
                device=device,
            )

            embeddings.append(self.encode_items(item_ids).cpu())

        return torch.cat(
            embeddings,
            dim=0,
        )
