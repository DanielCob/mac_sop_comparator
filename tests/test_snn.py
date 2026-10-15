import pytest
import snntorch as snn
import torch
from torch import nn

from mac_sop_comparator.models import AnnClassifier, BaseClassifier, SnnClassifier

TOPOLOGY = [1024, 128, 64, 2]


def make(topology=(8, 6, 2), T=5, **kw):
    return SnnClassifier(list(topology), T, kw.pop("beta", 0.95), kw.pop("threshold", 1.0), **kw)


def drive(model, weight, bias=0.0):
    """Give every Linear the same constant weights, so that activity is easy to predict."""
    with torch.no_grad():
        for fc in model.fcs:
            fc.weight.fill_(weight)
            fc.bias.fill_(bias)


def test_is_homologous_to_the_ann():
    snn_model, ann = make(TOPOLOGY, T=10), AnnClassifier(TOPOLOGY)
    assert isinstance(snn_model, BaseClassifier)
    assert [p.shape for p in snn_model.parameters()] == [p.shape for p in ann.parameters()]
    assert sum(p.numel() for p in snn_model.parameters()) == 139_392 + 194
    assert {type(m) for m in snn_model.modules()} == {SnnClassifier, nn.ModuleList, nn.Linear, snn.Leaky}  # no dropout, no batch norm
    assert [(l.in_features, l.out_features) for l in snn_model.layers()] == [(1024, 128), (128, 64), (64, 2)]
    assert len(snn_model.lifs) == 3  # one LIF per Linear, the output layer included


def test_forward_returns_integer_spike_counts_bounded_by_T():
    model = make(T=5)
    counts = model((torch.rand(16, 8) < 0.5).float())
    assert counts.shape == (16, 2)
    assert torch.equal(counts, counts.round()) and counts.min() >= 0 and counts.max() <= 5


def test_no_activity_gives_no_spikes():
    model = make()
    drive(model, 0.0)
    counts = model(torch.ones(3, 8))
    assert counts.sum() == 0
    assert all(s.sum() == 0 for layer, s in model.spike_counts().items() if layer > 0)


def test_strong_drive_makes_every_neuron_fire_at_every_step():
    model = make(T=6)
    drive(model, 1.0)  # input current 8 (> threshold) in the first layer, 6 in the second
    counts = model(torch.ones(2, 8))
    assert (counts == 6).all()
    spikes = model.spike_counts()
    assert {l: tuple(s.shape) for l, s in spikes.items()} == {0: (6, 2), 1: (6, 2), 2: (6, 2)}
    assert (spikes[0] == 8).all() and (spikes[1] == 6).all() and (spikes[2] == 2).all()  # spikes per step and sample


def test_spike_counts_match_the_output_and_the_encoded_input():
    torch.manual_seed(0)
    model = make((8, 6, 4, 2), T=7)
    x = (torch.rand(5, 8) < 0.4).float()
    out = model(x)
    spikes = model.spike_counts()
    assert torch.equal(spikes[len(model.fcs)].sum(dim=0).unsqueeze(1), out.sum(dim=1, keepdim=True))
    assert torch.equal(spikes[0], x.sum(dim=1).expand(7, 5))  # deterministic rate encoding


def test_accepts_a_ready_made_spike_train_and_checks_its_shape():
    model = make(T=4)
    train = torch.ones(4, 3, 8)
    assert model(train).shape == (3, 2)
    with pytest.raises(ValueError, match="T=4"):
        model(torch.ones(5, 3, 8))
    with pytest.raises(ValueError, match="expected"):
        model(torch.ones(8))


def test_membrane_sum_is_recorded_for_tie_breaking():
    model = make(T=5)
    model(torch.ones(4, 8))
    assert model.membrane_sum.shape == (4, 2)


def test_leak_and_reset_follow_the_lif_equations():
    # one neuron, constant current 1.5, threshold 1, beta 0.95, reset by subtraction
    model = SnnClassifier([1, 1], T=3, beta=0.95, threshold=1.0)
    drive(model, 1.5)
    model(torch.ones(1, 1))
    assert model.membrane_sum.item() == pytest.approx(1.5 + (0.95 * 1.5 + 1.5 - 1.0) + (0.95 * 1.925 + 1.5 - 1.0))


def test_zero_reset_differs_from_subtract_reset():
    a, b = make((4, 3, 2), T=8, reset="subtract"), make((4, 3, 2), T=8, reset="zero")
    drive(a, 0.6)
    drive(b, 0.6)
    x = torch.ones(2, 4)
    a(x), b(x)
    assert not torch.equal(a.membrane_sum, b.membrane_sum)


def test_deterministic_and_same_in_train_and_eval_mode():
    torch.manual_seed(0)
    model = make()
    x = (torch.rand(6, 8) < 0.5).float()
    model.train()
    a = model(x)
    model.eval()
    assert torch.equal(a, model(x))


def test_surrogate_gradient_reaches_every_layer():
    model = make((8, 6, 4, 2), T=6)
    drive(model, 0.3, 0.1)
    model(torch.ones(4, 8)).sum().backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(beta=0.0), "beta"),
        (dict(beta=1.5), "beta"),
        (dict(threshold=0.0), "threshold"),
        (dict(surrogate_gradient="sigmoid"), "surrogate"),
        (dict(reset="none"), "reset"),
        (dict(T=0), "T must be"),
    ],
)
def test_invalid_arguments(kwargs, message):
    with pytest.raises(ValueError, match=message):
        make(**kwargs)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_runs_on_mps():
    model = make().to("mps")
    out = model((torch.rand(4, 8, device="mps") < 0.5).float())
    assert out.device.type == "mps" and model.spike_counts()[0].device.type == "mps"
