import csv

import pytest

from mac_sop_comparator.comparison import Candidate, EfficiencyMetrics, ParetoAnalyzer, PointResult, ReportBuilder
from mac_sop_comparator.comparison import sensitivity
from mac_sop_comparator.config import ArchitectureParams
from mac_sop_comparator.profiling import OpReport

PARAMS = ArchitectureParams(weight_bits=32, e_mac=1.0, e_ac=0.2, e_upd=0.3, e_byte_von_neumann=53.0, e_byte_neuromorphic=1.3)
SIZES = [10, 5, 2]


def point(T=4, seed=42, hidden=(5,), ann_auc=0.74, snn_auc=0.70, sops=60.0):
    ann = OpReport("ann", macs=100)
    snn = OpReport("snn", sops=sops, state_updates=20.0, spikes_per_layer=[8.0, 3.0, 1.0])
    return PointResult(
        f"morgan10_h{'-'.join(map(str, hidden))}_T{T}_s{seed}", "morgan", 10, T, ann_auc, snn_auc, 0.6, 0.6, ann, snn,
        EfficiencyMetrics.compute(ann, snn, PARAMS, T, SIZES), PARAMS, hidden_layers=hidden, seed=seed,
    )  # fmt: skip


def cand(name, auc, cost, energy=None):
    return Candidate(name, name, "snn", name, auc, cost, cost if energy is None else energy)


# Frontier ----------------------------------------------------------------------------------------


def test_frontier_keeps_only_non_dominated_models_cheapest_first():
    a, b, c, d, e = cand("a", 0.70, 10), cand("b", 0.80, 20), cand("c", 0.75, 30), cand("d", 0.80, 20), cand("e", 0.60, 5)
    # c is beaten by b (better AUC, cheaper); equal models b and d do not dominate each other
    assert [x.candidate_id for x in ParetoAnalyzer.frontier([a, b, c, d, e])] == ["e", "a", "b", "d"]


def test_a_more_accurate_but_costlier_model_stays_and_so_does_a_cheaper_less_accurate_one():
    cheap, accurate = cand("cheap", 0.60, 1), cand("accurate", 0.90, 100)
    assert len(ParetoAnalyzer.frontier([cheap, accurate])) == 2


def test_same_cost_lower_auc_is_dominated():
    assert [c.candidate_id for c in ParetoAnalyzer.frontier([cand("x", 0.7, 10), cand("y", 0.8, 10)])] == ["y"]


def test_the_frontier_can_use_compute_cost_or_total_energy():
    # with compute cost p is on the frontier; with energy q beats it
    p, q = cand("p", 0.70, cost=1, energy=100), cand("q", 0.75, cost=2, energy=10)
    assert [c.candidate_id for c in ParetoAnalyzer.frontier([p, q], "cost")] == ["p", "q"]
    assert [c.candidate_id for c in ParetoAnalyzer.frontier([p, q], "energy")] == ["q"]
    with pytest.raises(ValueError, match="metric"):
        ParetoAnalyzer.frontier([p], "bytes")


# Candidates --------------------------------------------------------------------------------------


def test_each_point_gives_an_snn_and_the_ann_shared_across_t_is_listed_once():
    got = ParetoAnalyzer.candidates([point(T=4), point(T=8), point(T=4, seed=7)])
    assert [c.candidate_id for c in got] == [
        "morgan10_h5_T4_s42:ann", "morgan10_h5_T4_s42:snn", "morgan10_h5_T8_s42:snn",
        "morgan10_h5_T4_s7:ann", "morgan10_h5_T4_s7:snn",
    ]  # fmt: skip
    ann = got[0]
    assert (ann.auc, ann.cost) == (0.74, 100.0) and ann.energy == pytest.approx(100 + 53 * 468)
    assert got[1].energy == pytest.approx(18 + 1.3 * 52)


def test_another_topology_is_another_ann():
    got = ParetoAnalyzer.candidates([point(hidden=(5,)), point(hidden=(16,))])
    assert sum(c.architecture == "ann" for c in got) == 2


def test_no_results_is_an_error():
    with pytest.raises(ValueError, match="no results"):
        ParetoAnalyzer.candidates([])


def test_csv_marks_the_frontier_of_each_metric(tmp_path):
    results = [point(T=4, snn_auc=0.70), point(T=8, snn_auc=0.50, sops=600.0)]  # the T = 8 SNN is worse and costlier
    path = ParetoAnalyzer.write_csv(results, tmp_path / "pareto.csv")
    rows = {r["candidate_id"]: r for r in csv.DictReader(path.open())}
    assert rows["morgan10_h5_T4_s42:snn"]["on_frontier_cost"] == "True"
    assert rows["morgan10_h5_T8_s42:snn"]["on_frontier_cost"] == "False"
    assert rows["morgan10_h5_T8_s42:snn"]["on_frontier_energy"] == "False"
    assert rows["morgan10_h5_T4_s42:ann"]["on_frontier_cost"] == "True"  # the most accurate model


# Sensitivity -------------------------------------------------------------------------------------


def test_update_energy_sensitivity_by_hand():
    # C_SNN = 60*0.2 + 20*k*0.2 over C_ANN = 100
    got = sensitivity.cost_ratio_vs_update_energy(point(), [1.0, 1.5, 3.5])
    assert got == pytest.approx([0.16, 0.18, 0.26])


def test_the_default_update_range_is_the_loihi_one():
    assert sensitivity.UPD_RATIOS[0] == 1.0 and sensitivity.UPD_RATIOS[-1] == pytest.approx(3.5)


def test_byte_energy_sensitivity_by_hand():
    p = point()  # E_ANN = 100 + 53*468 = 24904, E_SNN = 18 + 1.3*52 = 85.6
    assert sensitivity.energy_ratio_vs_byte_energy(p, "von_neumann", [0.0, 53.0]) == pytest.approx([85.6 / 100, 85.6 / 24904])
    assert sensitivity.energy_ratio_vs_byte_energy(p, "neuromorphic", [0.0, 1.3]) == pytest.approx([18 / 24904, 85.6 / 24904])
    assert sensitivity.energy_ratio_vs_byte_energy(p, "neuromorphic", [1.3]) == pytest.approx([p.metrics.energy_ratio])
    with pytest.raises(ValueError, match="kind"):
        sensitivity.energy_ratio_vs_byte_energy(p, "gpu")


# Figures -----------------------------------------------------------------------------------------


def test_build_sweep_writes_all_the_consolidated_files(tmp_path):
    paths = ReportBuilder.build_sweep([point(T=4), point(T=8, snn_auc=0.72)], tmp_path / "sweep")
    assert set(paths) == {"points", "pareto", "update_sensitivity", "byte_sensitivity", "pareto_cost", "pareto_energy"}
    assert {p.name for p in paths.values()} == {
        "points.csv", "pareto.csv", "sensitivity_e_upd.png", "sensitivity_e_byte.png", "pareto_cost.png", "pareto_energy.png",
    }  # fmt: skip
    assert all(p.is_file() and p.stat().st_size > 0 for p in paths.values())


def test_a_single_point_still_draws_everything(tmp_path):
    assert len(ReportBuilder.build_sweep([point()], tmp_path)) == 6


def test_empty_sweep_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="no results"):
        ReportBuilder.build_sweep([], tmp_path)
