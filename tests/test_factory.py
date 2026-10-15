from pathlib import Path

import pytest
import torch
from torch import nn

from mac_sop_comparator.config import AnnConfig, ModelConfig, SnnConfig, load_config
from mac_sop_comparator.models import AnnClassifier, ModelFactory, SnnClassifier

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


def test_build_snn_with_the_config_parameters():
    cfg = CFG.model_copy(update={"snn": SnnConfig(T=7, beta=0.9, threshold=1.5, reset="zero")})
    model = ModelFactory.build(cfg, 1024, "snn")
    assert isinstance(model, SnnClassifier)
    assert (model.topology, model.T, model.beta, model.threshold, model.reset) == ([1024, 128, 64, 2], 7, 0.9, 1.5, "zero")


def test_ann_and_snn_are_homologous():
    """Same topology, same weight shapes, same parameter count: the only difference is the neuron model."""
    ann, snn = ModelFactory.build(CFG, 1024, "ann"), ModelFactory.build(CFG, 1024, "snn")
    assert ann.topology == snn.topology
    assert [n for n, _ in ann.named_parameters()] == [n for n, _ in snn.named_parameters()]
    assert [p.shape for p in ann.parameters()] == [p.shape for p in snn.parameters()]
    assert sum(p.numel() for p in ann.parameters()) == sum(p.numel() for p in snn.parameters())
    assert [l.weight.shape for l in ann.layers()] == [l.weight.shape for l in snn.layers()]


def test_same_seed_gives_both_models_the_same_initial_weights():
    torch.manual_seed(5)
    ann = ModelFactory.build(CFG, 1024, "ann")
    torch.manual_seed(5)
    snn = ModelFactory.build(CFG, 1024, "snn")
    assert all(torch.equal(a, b) for a, b in zip(ann.parameters(), snn.parameters()))


def test_ann_weights_can_be_loaded_into_the_snn():
    ann, snn = ModelFactory.build(CFG, 64, "ann"), ModelFactory.build(CFG, 64, "snn")
    result = snn.load_state_dict(ann.state_dict(), strict=False)
    assert not result.unexpected_keys  # every ANN weight finds its place in the SNN
    assert all(key.startswith("lifs.") for key in result.missing_keys)  # only the LIF neuron buffers are extra
    assert all(torch.equal(a, b) for a, b in zip(ann.parameters(), snn.parameters()))


def test_both_models_use_the_same_macs_as_synaptic_connections():
    from mac_sop_comparator.profiling import MacProfiler

    ann, snn = ModelFactory.build(CFG, 1024, "ann"), ModelFactory.build(CFG, 1024, "snn")
    assert MacProfiler(ann).measure().macs == MacProfiler(snn).measure().macs == 139_392  # N_MAC of the SOP bound


def test_unknown_kinds_and_bad_input_size_fail():
    with pytest.raises(ValueError, match="kind"):
        ModelFactory.build(CFG, 1024, "cnn")
    with pytest.raises(ValueError, match="n_inputs"):
        ModelFactory.build(CFG, 0, "ann")


def test_builds_from_the_shipped_config_with_the_prepared_feature_count():
    cfg = load_config(Path(__file__).parent.parent / "configs" / "tox21_sr_are.yaml")
    model = ModelFactory.build(cfg.model, cfg.data.fingerprint.n_bits, "ann")
    assert sum(l.weight.numel() for l in model.layers()) == 139_392  # the MAC count of D-16
