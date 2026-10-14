"""Abstract classifier shared by the ANN and the SNN (same topology => homologous models)."""

from abc import ABC, abstractmethod
from collections.abc import Sequence

import torch
from torch import nn


class BaseClassifier(nn.Module, ABC):
    """A feed-forward classifier defined by its topology, e.g. [1024, 128, 64, 2].

    Takes plain arguments and does not import `config` (D-22).
    """

    def __init__(self, topology: Sequence[int]):
        super().__init__()
        topology = [int(n) for n in topology]
        if len(topology) < 2:
            raise ValueError(f"topology needs at least an input and an output size, got {topology}")
        if any(n < 1 for n in topology):
            raise ValueError(f"every layer size must be >= 1, got {topology}")
        self.topology: list[int] = topology

    @property
    def n_inputs(self) -> int:
        return self.topology[0]

    @property
    def n_outputs(self) -> int:
        return self.topology[-1]

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Class scores for a batch (ANN: logits; SNN: output spike counts)."""

    @abstractmethod
    def layers(self) -> list[nn.Module]:
        """The weighted layers in order (nn.Linear for now), for the profilers to walk."""
