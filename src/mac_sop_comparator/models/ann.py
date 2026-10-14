"""Dense multilayer perceptron (the ANN of the comparison)."""

from collections.abc import Sequence

import torch
from torch import nn

from mac_sop_comparator.models.base import BaseClassifier

ACTIVATIONS: dict[str, type[nn.Module]] = {"relu": nn.ReLU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid}


class AnnClassifier(BaseClassifier):
    """Fully connected layers with an activation between them; the output layer returns logits.

    No dropout and no batch norm (D-32), so that the SNN counterpart has exactly the same weights.
    """

    def __init__(self, topology: Sequence[int], activation: str = "relu"):
        super().__init__(topology)
        if activation not in ACTIVATIONS:
            raise ValueError(f"activation must be one of {sorted(ACTIVATIONS)}, got '{activation}'")
        self.activation_name = activation
        self.activation = ACTIVATIONS[activation]()
        self.fcs = nn.ModuleList(nn.Linear(n_in, n_out) for n_in, n_out in zip(self.topology[:-1], self.topology[1:]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for fc in self.fcs[:-1]:
            x = self.activation(fc(x))
        return self.fcs[-1](x)

    def layers(self) -> list[nn.Linear]:
        return list(self.fcs)
