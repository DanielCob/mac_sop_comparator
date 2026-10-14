from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.flop_counter import FlopCounterMode

from mac_sop_comparator.config import load_config
from mac_sop_comparator.data import MolecularDataset
from mac_sop_comparator.models import AnnClassifier, BaseClassifier, ModelFactory
from mac_sop_comparator.profiling import MacProfiler, OpProfiler


def flop_counter_macs(model, x) -> int:
    """MACs according to PyTorch's own FLOP counter (FLOPs = 2 * MACs), per sample."""
    with FlopCounterMode(display=False) as counter:
        model(x)
    assert counter.get_total_flops() % (2 * len(x)) == 0
    return counter.get_total_flops() // (2 * len(x))


def test_known_topology():
    report = MacProfiler(AnnClassifier([1024, 128, 64, 2])).measure()
    assert report.architecture == "ann"
    assert report.macs == 1024 * 128 + 128 * 64 + 64 * 2 == 139_392
    assert report.biases == 128 + 64 + 2 == 194
    assert report.sops is None and report.state_updates is None


@pytest.mark.parametrize("topology", [[1024, 128, 64, 2], [166, 128, 64, 2], [10, 2], [2048, 256, 128, 64, 2]])
def test_matches_pytorch_flop_counter(topology):
    model = AnnClassifier(topology)
    macs = MacProfiler(model).measure().macs
    assert macs == flop_counter_macs(model, torch.rand(7, topology[0]))


def test_does_not_depend_on_the_data_or_the_weights():
    model = AnnClassifier([16, 8, 2])
    a = MacProfiler(model).measure()
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
    zeros = MolecularDataset(np.zeros((5, 16)), np.zeros(5))
    assert MacProfiler(model).measure(zeros) == a


def test_activation_choice_does_not_change_the_count():
    assert MacProfiler(AnnClassifier([16, 8, 2], "tanh")).measure() == MacProfiler(AnnClassifier([16, 8, 2], "relu")).measure()


def test_layer_without_bias_reports_fewer_biases():
    model = AnnClassifier([16, 8, 2])
    model.fcs[0].bias = None
    assert MacProfiler(model).measure().biases == 2


class ConvNet(BaseClassifier):
    """Conv1d -> Linear, to exercise the Conv1d branch (D-12)."""

    def __init__(self, length=20, channels=4, kernel=3):
        super().__init__([length, channels, 2])
        self.conv = nn.Conv1d(1, channels, kernel)
        self.fc = nn.Linear(channels * (length - kernel + 1), 2)

    def forward(self, x):
        return self.fc(torch.relu(self.conv(x.unsqueeze(1))).flatten(1))

    def layers(self):
        return [self.conv, self.fc]


def test_conv1d_is_counted_with_its_output_length_and_matches_flop_counter():
    model = ConvNet()
    data = MolecularDataset(np.random.default_rng(0).random((3, 20)), np.zeros(3))
    report = MacProfiler(model).measure(data)
    out_len = 20 - 3 + 1
    assert report.macs == out_len * (4 * 1 * 3) + 4 * out_len * 2
    assert report.macs == flop_counter_macs(model, torch.rand(5, 20))
    assert report.biases == 4 + 2


def test_conv1d_needs_data_and_restores_the_model_state():
    model = ConvNet()
    with pytest.raises(ValueError, match="needs data"):
        MacProfiler(model).measure()
    model.train()
    MacProfiler(model).measure(MolecularDataset(np.zeros((2, 20)), np.zeros(2)))
    assert model.training and not model.conv._forward_hooks


def test_unsupported_layer():
    class Odd(BaseClassifier):
        def __init__(self):
            super().__init__([4, 2])
            self.f = nn.Bilinear(4, 4, 2)

        def forward(self, x):
            return self.f(x, x)

        def layers(self):
            return [self.f]

    with pytest.raises(TypeError, match="supports nn.Linear and nn.Conv1d"):
        MacProfiler(Odd()).measure()


def test_is_an_op_profiler():
    assert issubclass(MacProfiler, OpProfiler)
    with pytest.raises(TypeError):
        OpProfiler(AnnClassifier([4, 2]))


def test_shipped_config_gives_139392_macs():
    cfg = load_config(Path(__file__).parent.parent / "configs" / "tox21_sr_are.yaml")
    model = ModelFactory.build(cfg.model, cfg.data.fingerprint.n_bits, "ann")
    assert MacProfiler(model).measure().macs == 139_392
