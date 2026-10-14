import pytest
import torch
from torch import nn

from mac_sop_comparator.models import BaseClassifier


class Tiny(BaseClassifier):
    """Minimal concrete model used only to exercise the base class."""

    def __init__(self, topology):
        super().__init__(topology)
        self.fcs = nn.ModuleList(nn.Linear(a, b) for a, b in zip(self.topology[:-1], self.topology[1:]))

    def forward(self, x):
        for fc in self.fcs:
            x = fc(x)
        return x

    def layers(self):
        return list(self.fcs)


def test_cannot_instantiate_the_abstract_class():
    with pytest.raises(TypeError):
        BaseClassifier([4, 2])


def test_subclass_must_implement_forward_and_layers():
    class NoLayers(BaseClassifier):
        def forward(self, x):
            return x

    with pytest.raises(TypeError):
        NoLayers([4, 2])


def test_topology_properties():
    model = Tiny((1024, 128, 64, 2))
    assert model.topology == [1024, 128, 64, 2]
    assert (model.n_inputs, model.n_outputs) == (1024, 2)
    assert isinstance(model, nn.Module)


def test_layers_and_forward():
    model = Tiny([8, 4, 2])
    assert [(l.in_features, l.out_features) for l in model.layers()] == [(8, 4), (4, 2)]
    assert model(torch.zeros(3, 8)).shape == (3, 2)


@pytest.mark.parametrize("bad", [[], [5], [8, 0, 2], [8, -1, 2]])
def test_invalid_topologies(bad):
    with pytest.raises(ValueError, match="topology|layer size"):
        Tiny(bad)
