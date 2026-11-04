import math

import pytest

from mac_sop_comparator.comparison import EfficiencyMetrics, MetricSet, critical_rate
from mac_sop_comparator.config import ArchitectureParams
from mac_sop_comparator.profiling import OpReport


def params(e_mac=1.0, e_ac=0.2, e_upd=0.3):
    return ArchitectureParams(weight_bits=32, e_mac=e_mac, e_ac=e_ac, e_upd=e_upd, e_byte_von_neumann=53.0, e_byte_neuromorphic=1.3)


def reports(macs=100, sops=60.0, upd=20.0, spikes=(8.0, 3.0, 1.0)):
    return OpReport("ann", macs=macs), OpReport("snn", sops=sops, state_updates=upd, spikes_per_layer=list(spikes))


SIZES = [10, 5, 2]


def test_hand_computed_values():
    # N_MAC=100, N_SOP=60, N_UPD=20, T=4: C_ANN = 100, C_SNN = 60*0.2 + 20*0.3 = 18, r = 60/400
    m = EfficiencyMetrics.compute(*reports(), params(), T=4, layer_sizes=SIZES)
    assert isinstance(m, MetricSet)
    assert (m.n_mac, m.n_sop, m.n_upd, m.T) == (100, 60.0, 20.0, 4)
    assert m.r == pytest.approx(0.15)
    assert m.cost_ann == pytest.approx(100.0)
    assert m.cost_snn == pytest.approx(18.0)
    assert m.cost_ratio == pytest.approx(0.18)
    assert m.ops_ratio == pytest.approx(0.8)
    assert m.r_critical == pytest.approx((100 - 6) / (4 * 0.2 * 100))  # 1.175
    assert m.snn_cheaper


def test_rate_per_layer_is_spikes_over_neurons_and_steps():
    m = EfficiencyMetrics.compute(*reports(), params(), T=4, layer_sizes=SIZES)
    assert m.rate_per_layer == pytest.approx([8 / 40, 3 / 20, 1 / 8])


def test_design_topology_numbers():
    # [1024, 128, 64, 2], T=10, illustrative energies: the figures quoted for the supervisor
    macs = 1024 * 128 + 128 * 64 + 64 * 2
    ann, snn = reports(macs=macs, sops=40_202.0, upd=1_940.0, spikes=(265.0, 96.0, 52.0, 4.0))
    m = EfficiencyMetrics.compute(ann, snn, params(), T=10, layer_sizes=[1024, 128, 64, 2])
    assert macs == 139_392
    assert m.r == pytest.approx(40_202 / (10 * 139_392))
    assert m.r_critical == pytest.approx((139_392 - 0.3 * 1_940) / (10 * 0.2 * 139_392))
    assert m.r_critical == pytest.approx(0.4979, abs=1e-4)
    assert m.cost_ratio == pytest.approx((40_202 * 0.2 + 1_940 * 0.3) / 139_392)
    assert m.r < m.r_critical and m.snn_cheaper


def test_costs_are_equal_exactly_at_the_critical_rate():
    p = params()
    r_star = critical_rate(p, n_mac=100, n_upd=20, T=4)
    ann, snn = reports(sops=r_star * 4 * 100)
    m = EfficiencyMetrics.compute(ann, snn, p, T=4, layer_sizes=SIZES)
    assert m.cost_snn == pytest.approx(m.cost_ann)
    assert m.r == pytest.approx(r_star)
    assert not m.snn_cheaper  # a tie is not cheaper


def test_snn_loses_when_it_is_more_active_than_critical():
    ann, snn = reports(sops=0.5 * 4 * 100 * 2.5)  # r = 1.25 > r* = 1.175
    m = EfficiencyMetrics.compute(ann, snn, params(), T=4, layer_sizes=SIZES)
    assert m.r > m.r_critical
    assert not m.snn_cheaper and m.cost_ratio > 1


def test_critical_rate_is_negative_when_state_updates_dominate():
    # state updates alone cost 1.0*200 > the whole ANN (100): no activity level makes the SNN cheaper
    r_star = critical_rate(params(e_upd=1.0), n_mac=100, n_upd=200, T=4)
    assert r_star == pytest.approx(-100 / (4 * 0.2 * 100))
    ann, snn = reports(sops=0.0, upd=200.0)
    m = EfficiencyMetrics.compute(ann, snn, params(e_upd=1.0), T=4, layer_sizes=SIZES)
    assert m.r_critical < 0 and not m.snn_cheaper


def test_critical_rate_with_free_synaptic_operations():
    assert critical_rate(params(e_ac=0.0), n_mac=100, n_upd=20, T=4) == math.inf
    assert critical_rate(params(e_ac=0.0, e_upd=1.0), n_mac=100, n_upd=200, T=4) == -math.inf


def test_critical_rate_scales_inversely_with_t():
    assert critical_rate(params(), 100, 0, 8) == pytest.approx(critical_rate(params(), 100, 0, 4) / 2)


def test_to_dict_is_json_ready():
    d = EfficiencyMetrics.compute(*reports(), params(), T=4, layer_sizes=SIZES).to_dict()
    assert set(d) == {
        "n_mac", "n_sop", "n_upd", "T", "r", "rate_per_layer", "cost_ann", "cost_snn",
        "cost_ratio", "ops_ratio", "r_critical", "snn_cheaper",
        "bytes_ann", "bytes_snn", "ai_ann", "ai_snn", "energy_ann", "energy_snn", "energy_ratio",
    }  # fmt: skip


@pytest.mark.parametrize(
    "ann, snn, T, sizes, message",
    [
        (OpReport("snn", sops=1.0, state_updates=1.0, spikes_per_layer=[1.0, 1.0, 1.0]), None, 4, SIZES, "expected an ANN"),
        (None, OpReport("ann", macs=1), 4, SIZES, "expected an ANN"),
        (None, None, 0, SIZES, "T must be"),
        (None, None, 4, [10, 5], "layer sizes"),
        (OpReport("ann", macs=0), None, 4, SIZES, "0 MACs"),
    ],
)  # fmt: skip
def test_invalid_inputs(ann, snn, T, sizes, message):
    default_ann, default_snn = reports()
    with pytest.raises(ValueError, match=message):
        EfficiencyMetrics.compute(ann or default_ann, snn or default_snn, params(), T=T, layer_sizes=sizes)


def test_memory_terms_by_hand():
    # layers [10, 5, 2]: ANN P = 100 MACs, A = 17, b_w = b_a = 4 -> 4*100 + 4*17 = 468 B (von Neumann)
    # SNN spikes [8, 3, 1]: events S_0 + S_1 = 11 -> 4*11 + 4*2 = 52 B (neuromorphic)
    m = EfficiencyMetrics.compute(*reports(), params(), T=4, layer_sizes=SIZES)
    assert (m.bytes_ann, m.bytes_snn) == (468.0, 52.0)
    assert m.ai_ann == pytest.approx(100 / 468)
    assert m.ai_snn == pytest.approx((60 + 20) / 52)
    assert m.energy_ann == pytest.approx(100 + 53 * 468)  # cost_ann + e_byte_von_neumann * bytes
    assert m.energy_snn == pytest.approx(18 + 1.3 * 52)  # cost_snn + e_byte_neuromorphic * bytes
    assert m.energy_ratio == pytest.approx(m.energy_snn / m.energy_ann)
