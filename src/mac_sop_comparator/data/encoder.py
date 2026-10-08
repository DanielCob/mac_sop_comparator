"""Spike encoding of binary fingerprints."""

import torch

from mac_sop_comparator.config import SnnConfig


class SpikeEncoder:
    """Deterministic rate encoding: a bit equal to 1 spikes at every time step, a 0 never does.

    The input spike train replicates the fingerprint T times, so it has no randomness.
    """

    def __init__(self, T: int):
        if T < 1:
            raise ValueError(f"T must be >= 1, got {T}")
        self.T = T

    @classmethod
    def from_config(cls, config: SnnConfig) -> "SpikeEncoder":
        return cls(config.T)

    def rate_encode(self, x: torch.Tensor) -> torch.Tensor:
        """(F,) -> (T, F) and (B, F) -> (T, B, F); same device, float32, values 0 or 1."""
        if x.ndim not in (1, 2):
            raise ValueError(f"x must have shape (F,) or (B, F), got {tuple(x.shape)}")
        if not torch.all((x == 0) | (x == 1)):
            raise ValueError("x must be binary (only 0 and 1) for deterministic rate encoding")
        return x.to(torch.float32).unsqueeze(0).repeat(self.T, *([1] * x.ndim))
