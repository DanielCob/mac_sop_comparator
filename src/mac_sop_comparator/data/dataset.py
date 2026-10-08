"""In-memory molecular dataset and mini-batch iteration."""

from collections.abc import Iterator

import numpy as np
import torch


class MolecularDataset:
    """Fingerprints `X` (n, F) as float32 zeros/ones and integer labels `y` (n,)."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D (samples, features), got shape {X.shape}")
        if y.shape != (len(X),):
            raise ValueError(f"y must have shape ({len(X)},), got {y.shape}")
        self.X, self.y = X, y

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i: int) -> tuple[np.ndarray, int]:
        return self.X[i], int(self.y[i])

    @property
    def n_features(self) -> int:
        return self.X.shape[1]

    def sparsity(self) -> float:
        """Fraction of zeros in X."""
        return float(1.0 - self.X.mean()) if self.X.size else 0.0

    def subset(self, indices: np.ndarray) -> "MolecularDataset":
        return MolecularDataset(self.X[indices], self.y[indices])


def batches(
    dataset: MolecularDataset,
    batch_size: int,
    shuffle: bool = False,
    generator: torch.Generator | None = None,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield (x, y) CPU tensors of at most `batch_size` samples; the last batch may be smaller.

    With `shuffle=True` the order comes from `generator`, which advances on every call: seed it
    once and each epoch gets a different, reproducible order.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if shuffle and generator is None:
        raise ValueError("shuffle=True requires a seeded torch.Generator (reproducibility)")
    n = len(dataset)
    order = torch.randperm(n, generator=generator).numpy() if shuffle else np.arange(n)
    for start in range(0, n, batch_size):
        idx = order[start : start + batch_size]
        yield torch.from_numpy(dataset.X[idx]), torch.from_numpy(dataset.y[idx])
