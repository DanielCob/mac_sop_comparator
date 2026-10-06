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


def set_num_threads(n_threads: int) -> None:
    """Fix the number of CPU threads used by PyTorch."""
    if n_threads < 1:
        raise ValueError(f"experiment.n_threads must be >= 1, got {n_threads}")
    os.environ["OMP_NUM_THREADS"] = str(n_threads)
    torch.set_num_threads(n_threads)
