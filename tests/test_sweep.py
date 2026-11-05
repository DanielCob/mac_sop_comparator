import csv
import json

import pytest
from conftest import CONFIG

from mac_sop_comparator.config import parse_config
from mac_sop_comparator.experiments import sweep as sweep_module
from mac_sop_comparator.experiments.layout import ResultsLayout
from mac_sop_comparator.experiments.runner import ExperimentRunner
from mac_sop_comparator.experiments.stages import ann_cache_key
from mac_sop_comparator.experiments.sweep import SweepManager


def make(sweep=None, **overrides):
    raw = {**CONFIG, "experiment": {**CONFIG["experiment"], "device": "cpu"}, "training": {**CONFIG["training"], "epochs": 2}}
    raw.update(overrides)
    if sweep is not None:
        raw["sweep"] = sweep
    return parse_config(raw)


def ids(manager, results="out"):
    return [ResultsLayout(results, c).point_id for c in manager.configs()]


# Points (no training) -----------------------------------------------------------------------------


def test_a_config_without_sweep_is_refused():
    with pytest.raises(ValueError, match="no 'sweep' section"):
        SweepManager(make())


def test_cartesian_product_is_ordered_so_that_an_ann_is_shared_by_consecutive_points():
    m = SweepManager(make({"T": [4, 8], "fingerprint": ["maccs", "morgan"], "n_bits": [1024, 2048], "hidden_layers": [[8], [16, 8]]}))
    got = ids(m)
    assert len(got) == 2 * 3 * 2  # T x (maccs + 2 morgan) x topologies
    assert got[:4] == ["maccs_h8_T4_s42", "maccs_h8_T8_s42", "maccs_h16-8_T4_s42", "maccs_h16-8_T8_s42"]
    assert got[4] == "morgan1024_r2_h8_T4_s42" and got[-1] == "morgan2048_r2_h16-8_T8_s42"


def test_n_bits_only_multiplies_morgan_and_maccs_appears_once():
    m = SweepManager(make({"T": [4], "fingerprint": ["maccs", "morgan"], "n_bits": [64, 128]}))
    assert ids(m) == ["maccs_h8_T4_s42", "morgan64_r2_h8_T4_s42", "morgan128_r2_h8_T4_s42"]


def test_without_extra_axes_the_base_topology_seed_and_bits_are_used():
    m = SweepManager(make({"T": [4, 8], "fingerprint": ["morgan"]}))
    assert ids(m) == ["morgan1024_r2_h8_T4_s42", "morgan1024_r2_h8_T8_s42"]


def test_seeds_are_an_axis_and_change_the_data_and_the_point():
    m = SweepManager(make({"T": [4], "fingerprint": ["morgan"], "seeds": [1, 2]}))
    configs = m.configs()
    assert [c.experiment.seed for c in configs] == [1, 2]
    assert ResultsLayout("out", configs[0]).data_dir != ResultsLayout("out", configs[1]).data_dir
    assert ids(m) == ["morgan1024_r2_h8_T4_s1", "morgan1024_r2_h8_T4_s2"]


def test_repeated_values_do_not_repeat_points():
    m = SweepManager(make({"T": [4, 4], "fingerprint": ["morgan", "morgan"], "seeds": [1, 1]}))
    assert len(m.configs()) == 1


def test_more_than_max_points_is_an_error_not_a_cut():
    m = SweepManager(make({"T": [2, 4, 8], "fingerprint": ["maccs", "morgan"], "max_points": 5}))
    with pytest.raises(ValueError, match="6 points, more than max_points = 5"):
        m.configs()


def test_each_point_config_changes_only_its_axes():
    base = make({"T": [8], "fingerprint": ["maccs"], "hidden_layers": [[16]], "seeds": [3]})
    (c,) = SweepManager(base).configs()
    assert (c.data.fingerprint.type, c.model.hidden_layers, c.model.snn.T, c.experiment.seed) == ("maccs", [16], 8, 3)
    assert (c.model.snn.beta, c.training, c.architecture) == (base.model.snn.beta, base.training, base.architecture)


def test_the_ann_cache_key_ignores_t_but_not_the_topology_or_the_seed():
    sweep = {"T": [4, 8], "fingerprint": ["morgan"], "hidden_layers": [[8], [16]], "seeds": [1, 2]}
    configs = {ResultsLayout("o", c).point_id: c for c in SweepManager(make(sweep)).configs()}

    def key(point_id):
        return ann_cache_key(configs[point_id])

    assert key("morgan1024_r2_h8_T4_s1") == key("morgan1024_r2_h8_T8_s1")
    assert key("morgan1024_r2_h8_T4_s1") != key("morgan1024_r2_h16_T4_s1")
    assert key("morgan1024_r2_h8_T4_s1") != key("morgan1024_r2_h8_T4_s2")


# Run (small real training) ---------------------------------------------------------------------------


def two_t_sweep():
    return make({"T": [2, 4], "fingerprint": ["morgan"]})


def test_the_second_t_copies_the_ann_and_trains_only_the_snn(workdir):
    result = SweepManager(two_t_sweep(), "out").run()
    first, second = "morgan1024_r2_h8_T2_s42", "morgan1024_r2_h8_T4_s42"
    assert [p.point_id for p in result.points] == [first, second]
    assert result.reused_ann == [second] and result.failed == {} and result.skipped == []
    root = workdir / "out" / "demo"
    assert (root / first / "ann.pt").read_bytes() == (root / second / "ann.pt").read_bytes()
    assert result.points[0].ann_auc == result.points[1].ann_auc
    assert (json.loads((root / first / "snn_metrics.json").read_text())["snn"]["T"], json.loads((root / second / "snn_metrics.json").read_text())["snn"]["T"]) == (2, 4)
    assert len(list((root / "data").iterdir())) == 1  # one prepared dataset for the whole sweep
    with open(result.paths["points"]) as f:
        rows = list(csv.DictReader(f))
    assert [r["point_id"] for r in rows] == [first, second] and rows[0]["seed"] == "42" and rows[0]["hidden_layers"] == "8"
    assert not (root / "sweep" / "failed.csv").exists()
    assert {f.name for f in (root / "sweep").iterdir()} == {
        "points.csv", "pareto.csv", "pareto_cost.png", "pareto_energy.png", "sensitivity_e_upd.png", "sensitivity_e_byte.png",
    }


def test_a_second_run_skips_the_points_with_a_report_and_force_reruns_them(workdir):
    SweepManager(two_t_sweep(), "out").run()
    again = SweepManager(two_t_sweep(), "out").run()
    assert len(again.skipped) == 2 and again.reused_ann == [] and len(again.points) == 2
    forced = SweepManager(two_t_sweep(), "out").run(force=True)
    assert forced.skipped == [] and len(forced.points) == 2


def test_a_failed_point_is_recorded_and_the_sweep_goes_on(workdir, monkeypatch):
    class Flaky(ExperimentRunner):
        def _train(self):
            if self.config.model.snn.T == 2:
                raise RuntimeError("boom")
            super()._train()

    monkeypatch.setattr(sweep_module, "ExperimentRunner", Flaky)
    result = SweepManager(two_t_sweep(), "out").run()
    assert [p.point_id for p in result.points] == ["morgan1024_r2_h8_T4_s42"]
    assert list(result.failed) == ["morgan1024_r2_h8_T2_s42"] and "boom" in result.failed["morgan1024_r2_h8_T2_s42"]
    assert result.reused_ann == []  # nothing to copy: the first point never trained an ANN
    with open(result.paths["failed"]) as f:
        assert list(csv.reader(f)) == [["point_id", "error"], ["morgan1024_r2_h8_T2_s42", result.failed["morgan1024_r2_h8_T2_s42"]]]


def test_a_clean_rerun_removes_a_stale_failed_csv(workdir):
    sweep_dir = workdir / "out" / "demo" / "sweep"
    sweep_dir.mkdir(parents=True)
    (sweep_dir / "failed.csv").write_text("point_id,error\nold,boom\n")
    SweepManager(make({"T": [2], "fingerprint": ["morgan"]}), "out").run()
    assert not (sweep_dir / "failed.csv").exists()
