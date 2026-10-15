"""Operation profilers: MACs for the ANN (here), SOPs for the SNN (phase 4)."""

from abc import ABC, abstractmethod

import torch
from torch import nn

from mac_sop_comparator.data import MolecularDataset
from mac_sop_comparator.data import batches
from mac_sop_comparator.models import BaseClassifier, SnnClassifier
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


PROFILE_BATCH_SIZE = 512  # does not affect the counts (D-31)


class SopProfiler(OpProfiler):
    """Synaptic operations of the SNN, averaged per sample over `data` (D-20, D-31).

    With S_l[t] the spikes emitted by layer l at step t (l = 0 is the encoded input) and N_l its neurons:
      SOPs            = sum_t sum_{l=0}^{L-1} S_l[t] * N_{l+1}   (spikes of the output layer have no fan-out)
      state updates   = T * sum_{l=1}^{L} N_l                    (every neuron, every step; biases included here)
      spikes_per_layer = [sum_t S_0, ..., sum_t S_L]             (L + 1 values, the output layer included)
    The biases are reported apart: they count neither as SOPs nor as MACs (D-16).
    """

    def __init__(self, model: SnnClassifier):
        if not isinstance(model, SnnClassifier):
            raise TypeError(f"SopProfiler needs an SnnClassifier, got {type(model).__name__}")
        super().__init__(model)

    @torch.no_grad()
    def measure(self, data: MolecularDataset | None = None) -> OpReport:
        if data is None or len(data) == 0:
            raise ValueError("counting SOPs needs data: the number of spikes depends on the molecules")
        model, topology = self.model, self.model.topology
        device = next(model.parameters()).device
        n_layers = len(topology) - 1
        spikes = torch.zeros(n_layers + 1, dtype=torch.float64)  # sum over samples of sum_t S_l[t]
        was_training = model.training
        model.eval()
        try:
            for x, _ in batches(data, PROFILE_BATCH_SIZE):
                model(x.to(device))
                for layer, per_step in model.spike_counts().items():  # (T, B)
                    spikes[layer] += per_step.cpu().double().sum()  # float64 only on the CPU (MPS has none)
        finally:
            model.train(was_training)
        spikes /= len(data)
        sops = sum(spikes[l].item() * topology[l + 1] for l in range(n_layers))
        state_updates = float(model.T * sum(topology[1:]))
        biases = sum(layer.bias.numel() for layer in model.layers() if layer.bias is not None)
        return OpReport(
            architecture="snn", sops=sops, state_updates=state_updates,
            spikes_per_layer=spikes.tolist(), biases=biases,
        )
