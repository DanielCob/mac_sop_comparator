"""Spiking counterpart of the ANN: the same Linear layers followed by leaky integrate-and-fire neurons."""

from collections.abc import Sequence

import snntorch as snn
import torch
from snntorch import surrogate
from torch import nn

from mac_sop_comparator.data import SpikeEncoder
from mac_sop_comparator.models.base import BaseClassifier

SURROGATES = {"arctan": surrogate.atan}
RESETS = ("subtract", "zero")


class SnnClassifier(BaseClassifier):
    """Each weighted layer is `nn.Linear` followed by a LIF neuron; the network is unrolled over T steps.

    The weights have exactly the shapes of `AnnClassifier(topology)` (homology). The output layer is also
    spiking, and the class scores are the spike counts of its neurons over the T steps.
    """

    def __init__(
        self,
        topology: Sequence[int],
        T: int,
        beta: float,
        threshold: float,
        surrogate_gradient: str = "arctan",
        reset: str = "subtract",
    ):
        super().__init__(topology)
        if not 0 < beta <= 1:
            raise ValueError(f"beta must be in (0, 1], got {beta}")
        if threshold <= 0:
            raise ValueError(f"threshold must be > 0, got {threshold}")
        if surrogate_gradient not in SURROGATES:
            raise ValueError(f"surrogate gradient must be one of {sorted(SURROGATES)}, got '{surrogate_gradient}'")
        if reset not in RESETS:
            raise ValueError(f"reset must be one of {RESETS}, got '{reset}'")
        self.encoder = SpikeEncoder(T)  # validates T >= 1
        self.T, self.beta, self.threshold, self.reset = T, beta, threshold, reset
        self.surrogate_name = surrogate_gradient
        self.fcs = nn.ModuleList(nn.Linear(n_in, n_out) for n_in, n_out in zip(self.topology[:-1], self.topology[1:]))
        self.lifs = nn.ModuleList(
            snn.Leaky(beta=beta, threshold=threshold, spike_grad=SURROGATES[surrogate_gradient](), reset_mechanism=reset)
            for _ in self.fcs
        )
        self._spikes: dict[int, torch.Tensor] = {}
        self.membrane_sum: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: binary fingerprints (B, F), encoded here into T steps, or a ready spike train (T, B, F).

        Returns the output spike counts (B, n_outputs). Side effects of the last call: `spike_counts()` and
        `membrane_sum` (output membrane potential summed over time, to break ties between counts).
        """
        spikes_in = self.encoder.rate_encode(x) if x.ndim == 2 else x
        if spikes_in.ndim != 3 or spikes_in.shape[0] != self.T:
            raise ValueError(f"expected (B, F) or a spike train (T={self.T}, B, F), got {tuple(x.shape)}")
        mems = [lif.init_leaky() for lif in self.lifs]
        per_layer: list[list[torch.Tensor]] = [[] for _ in range(len(self.fcs) + 1)]  # index 0 = input spikes
        out_counts, membrane_sum = 0, 0
        for t in range(self.T):
            spk = spikes_in[t]
            per_layer[0].append(spk.detach().sum(dim=1))
            for i, (fc, lif) in enumerate(zip(self.fcs, self.lifs), start=1):
                spk, mems[i - 1] = lif(fc(spk), mems[i - 1])
                per_layer[i].append(spk.detach().sum(dim=1))
            out_counts = out_counts + spk
            membrane_sum = membrane_sum + mems[-1].detach()
        self._spikes = {l: torch.stack(steps) for l, steps in enumerate(per_layer)}  # each (T, B)
        self.membrane_sum = membrane_sum
        return out_counts

    def positive_scores(self, outputs: torch.Tensor) -> torch.Tensor:
        """softmax of the spike counts with ties broken by the accumulated output membrane potential (D-19).

        For two classes softmax(counts)[1] = sigmoid(c1 - c0). Equal count differences are ordered by adding
        0.5 * softsign(m1 - m0) to the logit: that term lies in (-0.5, 0.5), so it never reorders samples whose
        count differences differ (integers >= 1 apart), and softsign, unlike tanh, does not saturate quickly.
        Must be called right after the `forward` that produced `outputs`. Returns float64.
        """
        if self.membrane_sum is None or self.membrane_sum.shape != outputs.shape:
            raise RuntimeError("positive_scores must be called right after the forward pass that produced `outputs`")
        counts, membrane = outputs.detach().cpu().double(), self.membrane_sum.cpu().double()  # float64: not on MPS
        gap = membrane[:, 1] - membrane[:, 0]
        return torch.sigmoid(counts[:, 1] - counts[:, 0] + 0.5 * gap / (1 + gap.abs()))

    def spike_counts(self) -> dict[int, torch.Tensor]:
        """Spikes emitted in the last forward pass: {layer: (T, B)}; layer 0 is the input, layer l >= 1 the
        LIF layer after the l-th Linear (the last one is the output)."""
        return self._spikes

    def layers(self) -> list[nn.Linear]:
        return list(self.fcs)
