from __future__ import annotations

import numpy as np
import pytest

from recsys.data.sampling import UniformNegativeSampler


def test_negative_sampler_never_returns_seen_items() -> None:
    user_ids = np.asarray(
        [
            0,
            0,
            1,
            1,
        ],
        dtype=np.int64,
    )

    item_ids = np.asarray(
        [
            0,
            1,
            1,
            2,
        ],
        dtype=np.int64,
    )

    sampler = UniformNegativeSampler(
        user_ids=user_ids,
        item_ids=item_ids,
        n_items=5,
        seed=42,
    )

    sampled_users = np.asarray(
        [
            0,
            0,
            0,
            1,
            1,
            1,
        ],
        dtype=np.int64,
    )

    negatives = sampler.sample(sampled_users)

    assert negatives.shape == sampled_users.shape
    assert np.issubdtype(negatives.dtype, np.integer)

    seen_by_user = {
        0: {0, 1},
        1: {1, 2},
    }

    for user_idx, item_idx in zip(
        sampled_users,
        negatives,
        strict=True,
    ):
        assert 0 <= int(item_idx) < 5
        assert int(item_idx) not in seen_by_user[int(user_idx)]


def test_dense_user_with_one_unseen_item_terminates_and_returns_it() -> None:
    """
    A near-full-catalog user must not enter an unbounded rejection loop.

    User 0 has seen items 0..998 from a 1,000-item catalog, leaving only
    item 999 as a valid negative. Every sampled negative must therefore
    be item 999.
    """

    n_items = 1_000

    user_ids = np.zeros(
        n_items - 1,
        dtype=np.int64,
    )

    item_ids = np.arange(
        n_items - 1,
        dtype=np.int64,
    )

    sampler = UniformNegativeSampler(
        user_ids=user_ids,
        item_ids=item_ids,
        n_items=n_items,
        seed=42,
    )

    sampled_users = np.zeros(
        128,
        dtype=np.int64,
    )

    negatives = sampler.sample(sampled_users)

    assert negatives.shape == sampled_users.shape

    np.testing.assert_array_equal(
        negatives,
        np.full(
            sampled_users.shape,
            n_items - 1,
            dtype=np.int64,
        ),
    )


def test_full_catalog_user_fails_instead_of_looping_forever() -> None:
    """
    A user with no valid negative item must fail explicitly.

    The assertion deliberately wraps construction and sampling so the
    implementation may reject the impossible user either eagerly during
    sampler construction or lazily when sample() is called.
    """

    n_items = 128

    user_ids = np.zeros(
        n_items,
        dtype=np.int64,
    )

    item_ids = np.arange(
        n_items,
        dtype=np.int64,
    )

    with pytest.raises(
        (ValueError, RuntimeError),
    ):
        sampler = UniformNegativeSampler(
            user_ids=user_ids,
            item_ids=item_ids,
            n_items=n_items,
            seed=42,
        )

        sampler.sample(
            np.asarray(
                [
                    0,
                ],
                dtype=np.int64,
            )
        )


def test_negative_sampler_is_reproducible_for_same_seed() -> None:
    user_ids = np.asarray(
        [
            0,
            0,
            1,
            1,
        ],
        dtype=np.int64,
    )

    item_ids = np.asarray(
        [
            0,
            1,
            1,
            2,
        ],
        dtype=np.int64,
    )

    sampled_users = np.asarray(
        [
            0,
            1,
            0,
            1,
            0,
            1,
            0,
            1,
        ],
        dtype=np.int64,
    )

    first_sampler = UniformNegativeSampler(
        user_ids=user_ids,
        item_ids=item_ids,
        n_items=10,
        seed=123,
    )

    second_sampler = UniformNegativeSampler(
        user_ids=user_ids,
        item_ids=item_ids,
        n_items=10,
        seed=123,
    )

    first = first_sampler.sample(sampled_users)

    second = second_sampler.sample(sampled_users)

    np.testing.assert_array_equal(
        first,
        second,
    )
