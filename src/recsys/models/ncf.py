from __future__ import annotations

import torch
from torch import nn


class GMF(nn.Module):
    """Generalized Matrix Factorization for implicit recommendation."""

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

        self.output = nn.Linear(
            embedding_dim,
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

        nn.init.xavier_uniform_(self.output.weight)

        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
    ) -> torch.Tensor:
        user_vectors = self.user_embedding(user_ids)

        item_vectors = self.item_embedding(item_ids)

        interaction = user_vectors * item_vectors

        return self.output(interaction).squeeze(-1)

    @torch.no_grad()
    def score_all(
        self,
        user_ids: torch.Tensor,
    ) -> torch.Tensor:
        user_vectors = self.user_embedding(user_ids)

        weighted_users = user_vectors * self.output.weight

        scores = weighted_users @ self.item_embedding.weight.T

        scores = scores + self.output.bias

        return scores


class MLPRecommender(nn.Module):
    """MLP-based Neural Collaborative Filtering model."""

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int = 64,
        hidden_layers: tuple[int, ...] = (
            128,
            64,
            32,
        ),
        item_score_batch_size: int = 2048,
    ) -> None:
        super().__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.item_score_batch_size = item_score_batch_size

        self.user_embedding = nn.Embedding(
            n_users,
            embedding_dim,
        )

        self.item_embedding = nn.Embedding(
            n_items,
            embedding_dim,
        )

        layers: list[nn.Module] = []

        input_dim = embedding_dim * 2

        for hidden_dim in hidden_layers:
            layers.extend(
                [
                    nn.Linear(
                        input_dim,
                        hidden_dim,
                    ),
                    nn.ReLU(),
                ]
            )

            input_dim = hidden_dim

        self.mlp = nn.Sequential(*layers)

        self.output = nn.Linear(
            input_dim,
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

        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)

                nn.init.zeros_(module.bias)

        nn.init.xavier_uniform_(self.output.weight)

        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
    ) -> torch.Tensor:
        user_vectors = self.user_embedding(user_ids)

        item_vectors = self.item_embedding(item_ids)

        features = torch.cat(
            [
                user_vectors,
                item_vectors,
            ],
            dim=1,
        )

        hidden = self.mlp(features)

        return self.output(hidden).squeeze(-1)

    @torch.no_grad()
    def score_all(
        self,
        user_ids: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = user_ids.size(0)

        user_vectors = self.user_embedding(user_ids)

        score_chunks = []

        for start in range(
            0,
            self.n_items,
            self.item_score_batch_size,
        ):
            end = min(
                start + self.item_score_batch_size,
                self.n_items,
            )

            item_ids = torch.arange(
                start,
                end,
                device=user_ids.device,
            )

            item_vectors = self.item_embedding(item_ids)

            chunk_size = item_vectors.size(0)

            expanded_users = user_vectors.unsqueeze(1).expand(
                -1,
                chunk_size,
                -1,
            )

            expanded_items = item_vectors.unsqueeze(0).expand(
                batch_size,
                -1,
                -1,
            )

            features = torch.cat(
                [
                    expanded_users,
                    expanded_items,
                ],
                dim=2,
            )

            hidden = self.mlp(
                features.reshape(
                    -1,
                    self.embedding_dim * 2,
                )
            )

            chunk_scores = self.output(hidden).reshape(
                batch_size,
                chunk_size,
            )

            score_chunks.append(chunk_scores)

        return torch.cat(
            score_chunks,
            dim=1,
        )


class NeuMF(nn.Module):
    """Neural Matrix Factorization combining GMF and MLP pathways."""

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int = 64,
        hidden_layers: tuple[int, ...] = (
            128,
            64,
            32,
        ),
        item_score_batch_size: int = 2048,
    ) -> None:
        super().__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.item_score_batch_size = item_score_batch_size

        self.gmf_user_embedding = nn.Embedding(
            n_users,
            embedding_dim,
        )

        self.gmf_item_embedding = nn.Embedding(
            n_items,
            embedding_dim,
        )

        self.mlp_user_embedding = nn.Embedding(
            n_users,
            embedding_dim,
        )

        self.mlp_item_embedding = nn.Embedding(
            n_items,
            embedding_dim,
        )

        layers: list[nn.Module] = []

        input_dim = embedding_dim * 2

        for hidden_dim in hidden_layers:
            layers.extend(
                [
                    nn.Linear(
                        input_dim,
                        hidden_dim,
                    ),
                    nn.ReLU(),
                ]
            )

            input_dim = hidden_dim

        self.mlp = nn.Sequential(*layers)

        self.output = nn.Linear(
            embedding_dim + input_dim,
            1,
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        embedding_layers = [
            self.gmf_user_embedding,
            self.gmf_item_embedding,
            self.mlp_user_embedding,
            self.mlp_item_embedding,
        ]

        for embedding in embedding_layers:
            nn.init.normal_(
                embedding.weight,
                mean=0.0,
                std=0.01,
            )

        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)

                nn.init.zeros_(module.bias)

        nn.init.xavier_uniform_(self.output.weight)

        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
    ) -> torch.Tensor:
        gmf_user = self.gmf_user_embedding(user_ids)

        gmf_item = self.gmf_item_embedding(item_ids)

        gmf_features = gmf_user * gmf_item

        mlp_user = self.mlp_user_embedding(user_ids)

        mlp_item = self.mlp_item_embedding(item_ids)

        mlp_features = torch.cat(
            [
                mlp_user,
                mlp_item,
            ],
            dim=1,
        )

        mlp_hidden = self.mlp(mlp_features)

        combined = torch.cat(
            [
                gmf_features,
                mlp_hidden,
            ],
            dim=1,
        )

        return self.output(combined).squeeze(-1)

    @torch.no_grad()
    def score_all(
        self,
        user_ids: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = user_ids.size(0)

        gmf_users = self.gmf_user_embedding(user_ids)

        mlp_users = self.mlp_user_embedding(user_ids)

        score_chunks = []

        for start in range(
            0,
            self.n_items,
            self.item_score_batch_size,
        ):
            end = min(
                start + self.item_score_batch_size,
                self.n_items,
            )

            item_ids = torch.arange(
                start,
                end,
                device=user_ids.device,
            )

            gmf_items = self.gmf_item_embedding(item_ids)

            mlp_items = self.mlp_item_embedding(item_ids)

            chunk_size = end - start

            gmf_features = gmf_users.unsqueeze(1) * gmf_items.unsqueeze(0)

            expanded_mlp_users = mlp_users.unsqueeze(1).expand(
                -1,
                chunk_size,
                -1,
            )

            expanded_mlp_items = mlp_items.unsqueeze(0).expand(
                batch_size,
                -1,
                -1,
            )

            mlp_features = torch.cat(
                [
                    expanded_mlp_users,
                    expanded_mlp_items,
                ],
                dim=2,
            )

            mlp_hidden = self.mlp(
                mlp_features.reshape(
                    -1,
                    self.embedding_dim * 2,
                )
            ).reshape(
                batch_size,
                chunk_size,
                -1,
            )

            combined = torch.cat(
                [
                    gmf_features,
                    mlp_hidden,
                ],
                dim=2,
            )

            chunk_scores = self.output(combined).squeeze(-1)

            score_chunks.append(chunk_scores)

        return torch.cat(
            score_chunks,
            dim=1,
        )


def initialize_neumf_from_pretrained(
    neumf: NeuMF,
    gmf: GMF,
    mlp: MLPRecommender,
    alpha: float = 0.5,
) -> None:
    """
    Initialize NeuMF from independently pretrained GMF and MLP models.
    """

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1.")

    with torch.no_grad():
        neumf.gmf_user_embedding.weight.copy_(gmf.user_embedding.weight)

        neumf.gmf_item_embedding.weight.copy_(gmf.item_embedding.weight)

        neumf.mlp_user_embedding.weight.copy_(mlp.user_embedding.weight)

        neumf.mlp_item_embedding.weight.copy_(mlp.item_embedding.weight)

        neumf.mlp.load_state_dict(mlp.mlp.state_dict())

        combined_output_weight = torch.cat(
            [
                alpha * gmf.output.weight,
                (1.0 - alpha) * mlp.output.weight,
            ],
            dim=1,
        )

        neumf.output.weight.copy_(combined_output_weight)

        combined_bias = alpha * gmf.output.bias + (1.0 - alpha) * mlp.output.bias

        neumf.output.bias.copy_(combined_bias)
