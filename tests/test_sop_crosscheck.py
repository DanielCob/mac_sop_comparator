"""P4.7: the SopProfiler against two independent ways of counting synaptic operations."""

from pathlib import Path

import numpy as np
import pytest
import torch

from mac_sop_comparator.data import MolecularDataset
from mac_sop_comparator.models import SnnClassifier, load_checkpoint
from mac_sop_comparator.profiling import MacProfiler, SopProfiler


def random_snn(topology, T, seed, beta=0.9, reset="subtract", gain=3.0):
    """A random SNN with enough activity for the test to be informative (weights scaled by `gain`)."""
    torch.manual_seed(seed)
    model = SnnClassifier(topology, T, beta, 1.0, reset=reset)
    with torch.no_grad():
        for p in model.parameters():
            p.mul_(gain)
    return model


def random_data(n, n_features, seed, density=0.3):
    rng = np.random.default_rng(seed)
    return MolecularDataset((rng.random((n, n_features)) < density).astype(np.float32), np.zeros(n))


def sops_with_forward_hooks(model, data) -> tuple[float, list[float]]:
    """Fig. 8 written by hand: observe the spikes with hooks and apply sum_t sum_l S_l[t] * fan-out."""
    n_layers = len(model.fcs)
    emitted = [torch.zeros(len(data), dtype=torch.float64) for _ in range(n_layers + 1)]  # per sample, summed over steps

    def observe_input(module, args):  # a hook must return None: any other value would replace the layer's input
        emitted[0] += args[0].double().sum(dim=1)

    def observe_layer(i):
        def hook(module, args, out):  # out = (spikes, membrane)
            emitted[i] += out[0].double().sum(dim=1)

        return hook

    handles = [model.fcs[0].register_forward_pre_hook(observe_input)]
    for i, lif in enumerate(model.lifs, start=1):
        handles.append(lif.register_forward_hook(observe_layer(i)))
    try:
        model.eval()
        with torch.no_grad():
            model(torch.from_numpy(data.X))
    finally:
        for h in handles:
            h.remove()
    fan_out = [fc.out_features for fc in model.fcs]
    per_sample = sum(emitted[l] * fan_out[l] for l in range(n_layers))  # the output layer has no fan-out
    return per_sample.mean().item(), [e.mean().item() for e in emitted]


def sops_with_numpy_lif(model, data) -> tuple[float, list[float]]:
    """No PyTorch dynamics at all: the LIF equations (subtract reset) and the SOP formula in plain NumPy."""
    W = [fc.weight.detach().numpy() for fc in model.fcs]
    b = [fc.bias.detach().numpy() for fc in model.fcs]
    X, beta, thr, T = data.X, np.float32(model.beta), np.float32(model.threshold), model.T
    n_layers = len(W)
    mem = [np.zeros((len(X), w.shape[0]), dtype=np.float32) for w in W]
    emitted = [np.zeros(len(X)) for _ in range(n_layers + 1)]
    for _ in range(T):
        spikes = X
        emitted[0] += spikes.sum(axis=1)
        for l in range(n_layers):
            reset = (mem[l] > thr).astype(np.float32)  # reset caused by the previous step's potential
            mem[l] = beta * mem[l] + (spikes @ W[l].T + b[l]) - reset * thr
            spikes = (mem[l] > thr).astype(np.float32)
            emitted[l + 1] += spikes.sum(axis=1)
    fan_out = [w.shape[0] for w in W]
    per_sample = sum(emitted[l] * fan_out[l] for l in range(n_layers))
    return float(per_sample.mean()), [float(e.mean()) for e in emitted]


CASES = [([16, 8, 2], 3, 0), ([32, 16, 8, 2], 10, 1), ([20, 12, 6, 2], 7, 2), ([64, 32, 2], 5, 3)]


@pytest.mark.parametrize("topology, T, seed", CASES)
@pytest.mark.parametrize("reset", ["subtract", "zero"])
def test_matches_a_hand_written_forward_hook_count(topology, T, seed, reset):
    model, data = random_snn(topology, T, seed, reset=reset), random_data(60, topology[0], seed)
    report = SopProfiler(model).measure(data)
    sops, spikes = sops_with_forward_hooks(model, data)
    assert sops > 0  # the case is not trivially silent
    assert report.sops == pytest.approx(sops, rel=1e-9)
    assert report.spikes_per_layer == pytest.approx(spikes, rel=1e-9)


@pytest.mark.parametrize("topology, T, seed", CASES)
def test_matches_a_pure_numpy_lif_reference(topology, T, seed):
    model, data = random_snn(topology, T, seed), random_data(60, topology[0], seed)
    report = SopProfiler(model).measure(data)
    sops, spikes = sops_with_numpy_lif(model, data)
    assert sops > 0
    assert report.sops == pytest.approx(sops, rel=1e-6)
    assert report.spikes_per_layer == pytest.approx(spikes, rel=1e-6)


@pytest.mark.parametrize("topology, T, seed", CASES)
def test_the_effective_firing_rate_is_a_fraction(topology, T, seed):
    """r = SOPs / (T * N_MAC): spikes are at most one per neuron and step, so 0 <= r <= 1."""
    model, data = random_snn(topology, T, seed), random_data(60, topology[0], seed)
    n_mac = sum(a * b for a, b in zip(topology[:-1], topology[1:]))
    r = SopProfiler(model).measure(data).sops / (T * n_mac)
    assert 0.0 <= r <= 1.0
    report = SopProfiler(model).measure(data)
    for spikes, n in zip(report.spikes_per_layer, topology):  # per layer: at most T spikes per neuron
        assert 0.0 <= spikes <= T * n


def test_the_firing_rate_reaches_both_extremes_of_its_range():
    topology, T = [1024, 128, 64, 2], 10
    n_mac = MacProfiler(random_snn(topology, T, 0)).measure().macs
    silent = random_snn(topology, T, 0)
    with torch.no_grad():
        for p in silent.parameters():
            p.zero_()
    silent_input = MolecularDataset(np.zeros((4, 1024), dtype=np.float32), np.zeros(4))
    assert SopProfiler(silent).measure(silent_input).sops == 0  # nothing in, nothing out: every term of the sum is 0
    full = random_snn(topology, T, 0)
    with torch.no_grad():
        for fc in full.fcs:
            fc.weight.fill_(1.0)
            fc.bias.zero_()
    ones = MolecularDataset(np.ones((4, 1024), dtype=np.float32), np.zeros(4))
    assert SopProfiler(full).measure(ones).sops == pytest.approx(T * n_mac)  # r = 1


RESULTS = Path(__file__).parent.parent / "results" / "tox21_sr_are"
CHECKPOINT = RESULTS / "morgan1024_r2_T10" / "snn.pt"
PREPARED = RESULTS / "data" / "morgan1024_r2"


@pytest.mark.skipif(not (CHECKPOINT.is_file() and PREPARED.is_dir()), reason="run prepare and train --model snn first")
def test_the_real_trained_snn_on_the_real_test_set():
    from mac_sop_comparator.data import load_prepared

    model, test = load_checkpoint(CHECKPOINT), load_prepared(PREPARED).partition("test")
    report = SopProfiler(model).measure(test)
    sops, spikes = sops_with_forward_hooks(model, test)
    assert report.sops == pytest.approx(sops, rel=1e-9) and report.spikes_per_layer == pytest.approx(spikes, rel=1e-9)
    assert report.sops == pytest.approx(sops_with_numpy_lif(model, test)[0], rel=1e-6)
    assert 0 < report.sops < model.T * 139_392
