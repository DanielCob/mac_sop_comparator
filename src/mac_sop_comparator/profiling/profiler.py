"""Operation profilers: MACs for the ANN (here), SOPs for the SNN (phase 4)."""

from abc import ABC, abstractmethod

import torch
from torch import nn

from mac_sop_comparator.data import MolecularDataset
from mac_sop_comparator.models import BaseClassifier
from mac_sop_comparator.profiling.report import OpReport


class OpProfiler(ABC):
    def __init__(self, model: BaseClassifier):
        self.model = model

    @abstractmethod
    def measure(self, data: MolecularDataset | None = None) -> OpReport:
        """Operations of one inference of `self.model` (averaged per sample when they depend on the data)."""


class MacProfiler(OpProfiler):
    """Counts multiply-accumulates of the weights only: no biases, no activations (D-16).

    A Linear layer costs `in * out` MACs, independent of the data. A Conv1d layer costs
    `output_length * weight.numel()`, so its output length is read with a forward hook from one sample.
    """

    def measure(self, data: MolecularDataset | None = None) -> OpReport:
        layers = self.model.layers()
        conv_lengths = self._conv_output_lengths(layers, data)
        macs = 0
        for layer in layers:
            if isinstance(layer, nn.Linear):
                macs += layer.weight.numel()
            elif isinstance(layer, nn.Conv1d):
                macs += conv_lengths[layer] * layer.weight.numel()
            else:
                raise TypeError(f"MacProfiler supports nn.Linear and nn.Conv1d, got {type(layer).__name__}")
        biases = sum(layer.bias.numel() for layer in layers if layer.bias is not None)
        return OpReport(architecture="ann", macs=macs, biases=biases)

    def _conv_output_lengths(self, layers: list[nn.Module], data: MolecularDataset | None) -> dict[nn.Module, int]:
        convs = [layer for layer in layers if isinstance(layer, nn.Conv1d)]
        if not convs:
            return {}
        if data is None or len(data) == 0:
            raise ValueError("counting the MACs of a Conv1d layer needs data (one sample) to get its output length")
        lengths: dict[nn.Module, int] = {}
        hooks = [c.register_forward_hook(lambda m, _in, out: lengths.__setitem__(m, out.shape[-1])) for c in convs]
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                self.model(torch.from_numpy(data.X[:1]).to(next(self.model.parameters()).device))
        finally:
            self.model.train(was_training)
            for hook in hooks:
                hook.remove()
        return lengths
