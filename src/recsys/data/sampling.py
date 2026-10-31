from __future__ import annotations

import numpy as np


class UniformNegativeSampler:
    """
    Uniformly sample items that are not present in a user's
    observed interaction history.

    Sampling is deterministic for a fixed seed and call order.

    For sparse users, rejection sampling is used for speed.
    If repeated collisions occur for a dense user, sampling
    falls back to explicitly constructing that user's set of
    available negative items.
    """

    def __init__(
        self,
        user_ids: np.ndarray,
        item_ids: np.ndarray,
        n_items: int,
        seed: int = 42,
    ) -> None:
        if not isinstance(n_items, int):
            raise TypeError("n_items must be an integer.")

        if n_items <= 0:
            raise ValueError("n_items must be > 0.")

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
            raise ValueError("interaction arrays cannot be empty.")

        if np.any(user_ids < 0):
            raise ValueError("user_ids cannot contain negative values.")

        if np.any((item_ids < 0) | (item_ids >= n_items)):
            raise ValueError(f"item_ids must be in the range [0, {n_items}).")

        self.n_items = n_items
        self.rng = np.random.default_rng(seed)

        interaction_keys = user_ids * n_items + item_ids

        self.positive_keys = np.sort(np.unique(interaction_keys))

        positive_users = self.positive_keys // self.n_items

        (
            unique_users,
            positive_counts,
        ) = np.unique(
            positive_users,
            return_counts=True,
        )

        self.full_catalog_users = unique_users[positive_counts >= self.n_items]

    def _validate_sample_users(
        self,
        user_ids: np.ndarray,
    ) -> np.ndarray:
        """Validate and normalize requested user IDs."""

        user_ids = np.asarray(
            user_ids,
            dtype=np.int64,
        )

        if user_ids.ndim != 1:
            raise ValueError("user_ids must be a 1D array.")

        if np.any(user_ids < 0):
            raise ValueError("user_ids cannot contain negative values.")

        if user_ids.size > 0 and self.full_catalog_users.size > 0:
            impossible_users = np.isin(
                user_ids,
                self.full_catalog_users,
            )

            if impossible_users.any():
                user = int(user_ids[impossible_users][0])

                raise ValueError(
                    "Cannot sample a negative item "
                    f"for user {user}: the user has "
                    "interacted with the entire catalog."
                )

        return user_ids

    def _is_positive(
        self,
        user_ids: np.ndarray,
        item_ids: np.ndarray,
    ) -> np.ndarray:
        """
        Return a boolean mask indicating whether each
        user-item pair is an observed positive.
        """

        keys = user_ids.astype(
            np.int64,
            copy=False,
        ) * self.n_items + item_ids.astype(
            np.int64,
            copy=False,
        )

        positions = np.searchsorted(
            self.positive_keys,
            keys,
        )

        valid_positions = positions < self.positive_keys.size

        collisions = np.zeros(
            keys.shape,
            dtype=bool,
        )

        collisions[valid_positions] = (
            self.positive_keys[positions[valid_positions]] == keys[valid_positions]
        )

        return collisions

    def _positive_items_for_user(
        self,
        user_id: int,
    ) -> np.ndarray:
        """
        Return sorted positive item IDs for one user.
        """

        start_key = user_id * self.n_items

        end_key = start_key + self.n_items

        start = np.searchsorted(
            self.positive_keys,
            start_key,
            side="left",
        )

        end = np.searchsorted(
            self.positive_keys,
            end_key,
            side="left",
        )

        return self.positive_keys[start:end] - start_key

    def _sample_exact_negative(
        self,
        user_id: int,
    ) -> int:
        """
        Sample uniformly from the exact set of unseen
        items for one user.

        Used as a fallback for dense users after repeated
        rejection-sampling collisions.
        """

        positive_items = self._positive_items_for_user(user_id)

        if positive_items.size >= self.n_items:
            raise ValueError(
                "Cannot sample a negative item "
                f"for user {user_id}: the user has "
                "interacted with the entire catalog."
            )

        all_items = np.arange(
            self.n_items,
            dtype=np.int64,
        )

        negative_items = np.setdiff1d(
            all_items,
            positive_items,
            assume_unique=True,
        )

        selected_index = self.rng.integers(
            low=0,
            high=negative_items.size,
        )

        return int(negative_items[selected_index])

    def sample(
        self,
        user_ids: np.ndarray,
    ) -> np.ndarray:
        """
        Sample one unseen item for every supplied user.

        Parameters
        ----------
        user_ids:
            1D array of encoded non-negative user IDs.

        Returns
        -------
        np.ndarray
            One negative item index per input user.
        """

        user_ids = self._validate_sample_users(user_ids)

        if user_ids.size == 0:
            return np.empty(
                0,
                dtype=np.int64,
            )

        negative_items = self.rng.integers(
            low=0,
            high=self.n_items,
            size=user_ids.size,
            dtype=np.int64,
        )

        collisions = self._is_positive(
            user_ids,
            negative_items,
        )

        max_rejection_rounds = 100
        rejection_round = 0

        while collisions.any() and rejection_round < max_rejection_rounds:
            collision_count = int(collisions.sum())

            negative_items[collisions] = self.rng.integers(
                low=0,
                high=self.n_items,
                size=collision_count,
                dtype=np.int64,
            )

            collisions = self._is_positive(
                user_ids,
                negative_items,
            )

            rejection_round += 1

        if collisions.any():
            unresolved_indices = np.flatnonzero(collisions)

            for index in unresolved_indices:
                negative_items[index] = self._sample_exact_negative(
                    user_id=int(user_ids[index])
                )

        final_collisions = self._is_positive(
            user_ids,
            negative_items,
        )

        if final_collisions.any():
            raise RuntimeError("Negative sampling produced an observed positive item.")

        return negative_items
