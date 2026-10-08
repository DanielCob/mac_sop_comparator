"""Reproducible stratified train/validation/test split (D-11)."""

import logging

import numpy as np
from sklearn.model_selection import train_test_split

from mac_sop_comparator.config import SplitConfig

logger = logging.getLogger(__name__)

PARTITIONS = ("train", "val", "test")


class Splitter:
    def __init__(self, train: float, val: float, test: float, seed: int):
        if abs(train + val + test - 1.0) > 1e-6:
            raise ValueError(f"train + val + test must sum to 1, got {train + val + test:g}")
        self.train, self.val, self.test, self.seed = train, val, test, seed

    @classmethod
    def from_config(cls, config: SplitConfig, seed: int) -> "Splitter":
        return cls(config.train, config.val, config.test, seed)

    def split(self, y: np.ndarray) -> dict[str, np.ndarray]:
        """Return sorted index arrays {"train", "val", "test"} stratified by the labels `y`."""
        y = np.asarray(y)
        idx = np.arange(len(y))
        try:
            train_idx, rest_idx = train_test_split(
                idx, test_size=self.val + self.test, stratify=y, random_state=self.seed
            )
            val_idx, test_idx = train_test_split(
                rest_idx, test_size=self.test / (self.val + self.test), stratify=y[rest_idx], random_state=self.seed
            )
        except ValueError as exc:
            raise ValueError(f"cannot split {len(y)} samples with class counts {np.bincount(y)}: {exc}") from exc
        parts = {"train": np.sort(train_idx), "val": np.sort(val_idx), "test": np.sort(test_idx)}
        logger.info(
            "split (seed %d): %s",
            self.seed,
            ", ".join(f"{k} {len(v)} ({y[v].mean():.3f} positive)" for k, v in parts.items()),
        )
        return parts
