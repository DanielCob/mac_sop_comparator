from pathlib import Path

import pytest
import torch
from torch import nn

from mac_sop_comparator.config import AnnConfig, ModelConfig, SnnConfig, load_config
from mac_sop_comparator.models import AnnClassifier, ModelFactory

CFG = ModelConfig(hidden_layers=[128, 64], n_outputs=2, snn=SnnConfig(T=10, beta=0.95, threshold=1.0))


def test_topology_is_input_hidden_output():
    assert ModelFactory.topology(CFG, 1024) == [1024, 128, 64, 2]
    assert ModelFactory.topology(CFG, 166) == [166, 128, 64, 2]  # maccs


def test_build_ann():
    model = ModelFactory.build(CFG, 1024, "ann")
    assert isinstance(model, AnnClassifier)
    assert model.topology == [1024, 128, 64, 2]
    assert isinstance(model.activation, nn.ReLU)


def test_activation_comes_from_the_config():
    cfg = CFG.model_copy(update={"ann": AnnConfig(activation="tanh")})
    assert isinstance(ModelFactory.build(cfg, 16, "ann").activation, nn.Tanh)


def test_each_build_gives_independent_weights():
    a, b = ModelFactory.build(CFG, 16, "ann"), ModelFactory.build(CFG, 16, "ann")
    assert not torch.equal(a.layers()[0].weight, b.layers()[0].weight)


def test_snn_is_not_available_yet_and_unknown_kinds_fail():
    with pytest.raises(NotImplementedError, match="phase 4"):
        ModelFactory.build(CFG, 1024, "snn")
    with pytest.raises(ValueError, match="kind"):
        ModelFactory.build(CFG, 1024, "cnn")
    with pytest.raises(ValueError, match="n_inputs"):
        ModelFactory.build(CFG, 0, "ann")


def test_builds_from_the_shipped_config_with_the_prepared_feature_count():
    cfg = load_config(Path(__file__).parent.parent / "configs" / "tox21_sr_are.yaml")
    model = ModelFactory.build(cfg.model, cfg.data.fingerprint.n_bits, "ann")
    assert sum(l.weight.numel() for l in model.layers()) == 139_392  # the MAC count of D-16
