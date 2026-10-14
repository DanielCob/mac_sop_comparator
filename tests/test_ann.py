import pytest
import torch
from torch import nn

from mac_sop_comparator.models import AnnClassifier, BaseClassifier

TOPOLOGY = [1024, 128, 64, 2]


def test_structure_follows_the_topology():
    model = AnnClassifier(TOPOLOGY)
    assert isinstance(model, BaseClassifier)
    assert [(l.in_features, l.out_features) for l in model.layers()] == [(1024, 128), (128, 64), (64, 2)]
    assert all(l.bias is not None for l in model.layers())


def test_parameter_count():
    # weights 139 392 (= MACs, D-16) + biases 194
    assert sum(p.numel() for p in AnnClassifier(TOPOLOGY).parameters()) == 139_392 + 194


def test_forward_returns_logits_of_the_right_shape():
    out = AnnClassifier(TOPOLOGY)(torch.rand(16, 1024))
    assert out.shape == (16, 2)


def test_output_layer_has_no_activation():
    model = AnnClassifier([4, 3, 2])
    with torch.no_grad():
        model.layers()[-1].weight.fill_(-1.0)
        model.layers()[-1].bias.fill_(-1.0)
    assert (model(torch.ones(5, 4)) < 0).all()  # a ReLU on the output would give zeros


def test_hidden_activation_is_applied_between_layers():
    model = AnnClassifier([2, 2, 1], "relu")
    with torch.no_grad():
        model.layers()[0].weight.copy_(torch.eye(2))
        model.layers()[0].bias.zero_()
        model.layers()[1].weight.fill_(1.0)
        model.layers()[1].bias.zero_()
    assert model(torch.tensor([[-3.0, 2.0]])).item() == pytest.approx(2.0)  # relu(-3) + relu(2)


@pytest.mark.parametrize("name, cls", [("relu", nn.ReLU), ("tanh", nn.Tanh), ("sigmoid", nn.Sigmoid)])
def test_activation_choice(name, cls):
    assert isinstance(AnnClassifier([4, 3, 2], name).activation, cls)


def test_unknown_activation():
    with pytest.raises(ValueError, match="activation"):
        AnnClassifier([4, 3, 2], "gelu")


def test_only_linear_layers_and_activation_no_dropout_or_batchnorm():
    kinds = {type(m) for m in AnnClassifier(TOPOLOGY).modules()}
    assert kinds == {AnnClassifier, nn.ModuleList, nn.Linear, nn.ReLU}


def test_same_seed_gives_same_weights():
    torch.manual_seed(1)
    a = AnnClassifier(TOPOLOGY)
    torch.manual_seed(1)
    b = AnnClassifier(TOPOLOGY)
    assert all(torch.equal(p, q) for p, q in zip(a.parameters(), b.parameters()))


def test_gradients_reach_every_layer():
    model = AnnClassifier([8, 6, 4, 2])
    model(torch.rand(5, 8)).sum().backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_runs_on_mps():
    model = AnnClassifier(TOPOLOGY).to("mps")
    assert model(torch.rand(16, 1024, device="mps")).device.type == "mps"
