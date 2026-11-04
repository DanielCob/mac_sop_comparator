import pytest

from mac_sop_comparator.config import ArchitectureParams
from mac_sop_comparator.profiling import OpReport, TrafficModel

BASE = dict(weight_bits=32, e_mac=1.0, e_ac=0.2, e_upd=0.3, e_byte_von_neumann=53.0, e_byte_neuromorphic=1.3)
SIZES = [10, 5, 2]
ANN = OpReport("ann", macs=100, biases=7)
SNN = OpReport("snn", sops=60.0, state_updates=20.0, spikes_per_layer=[8.0, 3.0, 1.0])


def model(**overrides):
    return TrafficModel(ArchitectureParams(**{**BASE, **overrides}), SIZES)


def test_ann_von_neumann_counts_weights_biases_and_every_layer():
    # b_w*(P + biases)/B_lote + b_a*A = 4*107 + 4*17
    assert model().bytes_accessed(ANN) == 4 * 107 + 4 * 17


def test_batch_shares_the_weights_but_not_the_activations():
    assert model(inference_batch=4).bytes_accessed(ANN) == 4 * 107 / 4 + 4 * 17


def test_bytes_follow_the_parameters():
    # 16-bit weights -> b_w = 2; b_a = 1
    assert model(weight_bits=16, b_a=1).bytes_accessed(ANN) == 2 * 107 + 1 * 17


def test_ann_without_biases_in_the_report():
    assert model().bytes_accessed(OpReport("ann", macs=100)) == 4 * 100 + 4 * 17


def test_snn_neuromorphic_counts_input_and_hidden_events_plus_outputs():
    # b_e*(S_0 + S_1) + b_o*N_L = 4*11 + 4*2; the output spikes (S_L) are not events that feed a layer
    assert model().bytes_accessed(SNN) == 52.0
    assert model(b_e=2, b_o=8).bytes_accessed(SNN) == 2 * 11 + 8 * 2


def test_snn_von_neumann_is_available_but_not_assigned():
    assert model().bytes_accessed(SNN, "von_neumann") == 4 * 60 + 2 * 4 * 20


def test_ann_in_the_neuromorphic_scenario_is_refused():
    with pytest.raises(ValueError, match="only evaluated in the von Neumann"):
        model().bytes_accessed(ANN, "neuromorphic")


def test_unknown_scenario_and_bad_inputs():
    with pytest.raises(ValueError, match="scenario"):
        model().bytes_accessed(ANN, "gpu")
    with pytest.raises(ValueError, match="spike counts"):
        TrafficModel(ArchitectureParams(**BASE), [10, 2]).bytes_accessed(SNN)
    with pytest.raises(ValueError, match="layer_sizes"):
        TrafficModel(ArchitectureParams(**BASE), [10])
    with pytest.raises(ValueError, match="bytes"):
        TrafficModel.intensity(5, 0)


def test_e_byte_is_the_one_of_the_assigned_scenario():
    assert (model().e_byte("ann"), model().e_byte("snn")) == (53.0, 1.3)


def test_annotate_fills_only_the_assigned_scenario_and_keeps_the_counts():
    ann, snn = model().annotate(ANN), model().annotate(SNN)
    assert ann.bytes_accessed == {"von_neumann": 496.0, "neuromorphic": None}
    assert snn.bytes_accessed == {"von_neumann": None, "neuromorphic": 52.0}
    assert (ann.macs, snn.sops) == (100, 60.0)
    assert ANN.bytes_accessed is None  # the original is untouched


def test_design_topology_numbers():
    # [1024, 128, 64, 2]: P = 139,392 weights + 194 biases, A = 1,218 -> 4*139,586 + 4*1,218 = 563,216 B
    sizes = [1024, 128, 64, 2]
    ann = OpReport("ann", macs=139_392, biases=194)
    assert TrafficModel(ArchitectureParams(**BASE), sizes).bytes_accessed(ann) == 563_216
