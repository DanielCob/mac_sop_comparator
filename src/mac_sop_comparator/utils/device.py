"""Compute device selection (MPS by default, CPU as fallback, D-28)."""

import logging

import torch

logger = logging.getLogger(__name__)

VALID_DEVICES = ("mps", "cpu")


def get_device(preferred: str = "mps") -> torch.device:
    """Return the requested device, falling back to CPU if MPS is unavailable."""
    if preferred not in VALID_DEVICES:
        raise ValueError(f"experiment.device must be one of {VALID_DEVICES}, got '{preferred}'")
    if preferred == "mps" and not torch.backends.mps.is_available():
        logger.warning("MPS is not available; falling back to CPU")
        return torch.device("cpu")
    return torch.device(preferred)
