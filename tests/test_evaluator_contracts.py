from __future__ import annotations

import numpy as np
import pytest
import torch

from recsys.evaluation.evaluator import SeenItemsCSR, evaluate_model


class _NonFiniteSeenItemModel:
    def eval(self):
        return self

    def score_all(self, user_ids: torch.Tensor) -> torch.Tensor:
        assert user_ids.numel() == 1
        return torch.tensor(
            [[float("nan"), 1.0]],
            dtype=torch.float32,
            device=user_ids.device,
        )


def test_raw_non_finite_score_is_rejected_before_seen_masking() -> None:
    seen_items = SeenItemsCSR(
        indptr=np.asarray([0, 1], dtype=np.int64),
        indices=np.asarray([0], dtype=np.int64),
    )

    with pytest.raises(
        RuntimeError,
        match="Model scores contain non-finite values before masking",
    ):
        evaluate_model(
            model=_NonFiniteSeenItemModel(),
            target_items=np.asarray([1], dtype=np.int64),
            seen_items=seen_items,
            n_users=1,
            batch_size=1,
            ks=(1,),
            device=torch.device("cpu"),
            desc="test",
        )
