from pathlib import Path

import pytest

from mac_sop_comparator.config import FingerprintConfig, parse_config
from mac_sop_comparator.experiments.layout import KINDS, ResultsLayout

RAW = {
    "experiment": {"name": "demo", "seed": 42},
    "data": {
        "dataset": "tox21",
        "task": "TASK",
        "fingerprint": {"type": "morgan", "n_bits": 1024, "radius": 2},
        "split": {"train": 0.8, "val": 0.1, "test": 0.1},
    },
    "model": {"hidden_layers": [8], "n_outputs": 2, "snn": {"T": 10, "beta": 0.9, "threshold": 1.0}},
    "training": {"optimizer": "adam", "lr": 1e-3, "batch_size": 16, "epochs": 1},
    "architecture": {"weight_bits": 32, "e_mac": 1, "e_ac": 0.2, "e_upd": 0.3, "e_byte_von_neumann": 53, "e_byte_neuromorphic": 1.3},
}


def layout(results_dir="results", **overrides):
    raw = {key: {**RAW[key], **overrides.get(key, {})} for key in RAW}
    return ResultsLayout(results_dir, parse_config(raw))


def test_documented_layout():
    lay = layout("results")
    point = Path("results/demo/morgan1024_r2_h8_T10_s42")
    assert lay.experiment_dir == Path("results/demo")
    assert lay.log_path == Path("results/demo/run.log")
    assert lay.data_dir == Path("results/demo/data/morgan1024_r2_s42")
    assert lay.point_id == "morgan1024_r2_h8_T10_s42" and lay.point_dir == point
    assert lay.report_dir == point / "report"
    assert lay.model_path("ann") == point / "ann.pt"
    assert lay.metrics_path("snn") == point / "snn_metrics.json"
    assert lay.curve_path("ann") == point / "ann_curve.png"
    assert lay.opreport_path("snn") == point / "opreport_snn.json"


def test_point_depends_on_the_fingerprint_and_t_but_data_only_on_the_fingerprint():
    base = layout()
    other_t = layout(model={"snn": {**RAW["model"]["snn"], "T": 4}})
    assert other_t.point_dir != base.point_dir and other_t.point_id == "morgan1024_r2_h8_T4_s42"
    assert other_t.data_dir == base.data_dir  # the data do not depend on T (D-25)
    maccs = layout(data={"fingerprint": {"type": "maccs", "n_bits": 1024, "radius": 2}})
    assert maccs.point_id == "maccs_h8_T10_s42" and maccs.data_dir.name == "maccs_s42"
    bits_2048 = layout(data={"fingerprint": {"type": "morgan", "n_bits": 2048, "radius": 2}})
    assert bits_2048.point_id == "morgan2048_r2_h8_T10_s42" and bits_2048.data_dir != base.data_dir


def test_seed_moves_the_point_and_the_data_but_epochs_and_energies_do_not():
    base = layout()
    other_seed = layout(experiment={"seed": 7})
    assert other_seed.point_id == "morgan1024_r2_h8_T10_s7" and other_seed.point_dir != base.point_dir
    assert other_seed.data_dir == Path("results/demo/data/morgan1024_r2_s7")  # the seed changes the split (D-44)
    same = layout(training={"epochs": 99}, architecture={"e_mac": 4.6, "e_ac": 0.9})
    assert same.point_dir == base.point_dir and same.data_dir == base.data_dir


def test_topology_is_part_of_the_point_id():
    assert layout(model={"hidden_layers": [128, 64]}).point_id == "morgan1024_r2_h128-64_T10_s42"
    assert layout(model={"hidden_layers": [16]}).point_dir != layout().point_dir


def test_sweep_and_analysis_dirs():
    assert layout().sweep_dir == Path("results/demo/sweep")
    assert layout().analysis_dir == Path("results/demo/analysis")


def test_an_explicit_fingerprint_selects_another_point_of_the_same_config():
    config = parse_config(RAW)
    lay = ResultsLayout("results", config, FingerprintConfig(type="morgan", n_bits=2048, radius=2))
    assert lay.point_id == "morgan2048_r2_h8_T10_s42"
    assert ResultsLayout("results", config).point_id == "morgan1024_r2_h8_T10_s42"


def test_results_dir_may_be_a_string_or_a_path(tmp_path):
    assert layout(str(tmp_path)).point_dir == layout(tmp_path).point_dir


@pytest.mark.parametrize("method", ["model_path", "metrics_path", "curve_path", "opreport_path"])
def test_unknown_model_kinds_are_rejected(method):
    with pytest.raises(ValueError, match="model kind"):
        getattr(layout(), method)("cnn")


def test_kinds():
    assert KINDS == ("ann", "snn")
