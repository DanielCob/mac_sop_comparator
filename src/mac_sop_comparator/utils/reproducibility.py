"""Seeds and thread count, so that runs are repeatable."""

import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch (CPU and MPS)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def set_num_threads(n_threads: int) -> int:
    """Fix the CPU threads used by PyTorch; 0 means all available cores. Returns the count.

    This only affects work that runs on the CPU. Tensor operations on MPS run on the GPU.
    """
    if n_threads < 0:
        raise ValueError(f"experiment.n_threads must be >= 0 (0 = all cores), got {n_threads}")
    n = n_threads or os.cpu_count() or 1
    torch.set_num_threads(n)
    return n
