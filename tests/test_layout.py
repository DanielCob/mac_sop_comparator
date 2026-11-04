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
    point = Path("results/demo/morgan1024_r2_T10")
    assert lay.experiment_dir == Path("results/demo")
    assert lay.log_path == Path("results/demo/run.log")
    assert lay.data_dir == Path("results/demo/data/morgan1024_r2")
    assert lay.point_id == "morgan1024_r2_T10" and lay.point_dir == point
    assert lay.report_dir == point / "report"
    assert lay.model_path("ann") == point / "ann.pt"
    assert lay.metrics_path("snn") == point / "snn_metrics.json"
    assert lay.curve_path("ann") == point / "ann_curve.png"
    assert lay.opreport_path("snn") == point / "opreport_snn.json"


def test_point_depends_on_the_fingerprint_and_t_but_data_only_on_the_fingerprint():
    base = layout()
    other_t = layout(model={"snn": {**RAW["model"]["snn"], "T": 4}})
    assert other_t.point_dir != base.point_dir and other_t.point_id == "morgan1024_r2_T4"
    assert other_t.data_dir == base.data_dir  # the data do not depend on T (D-25)
    maccs = layout(data={"fingerprint": {"type": "maccs", "n_bits": 1024, "radius": 2}})
    assert maccs.point_id == "maccs_T10" and maccs.data_dir.name == "maccs"
    bits_2048 = layout(data={"fingerprint": {"type": "morgan", "n_bits": 2048, "radius": 2}})
    assert bits_2048.point_id == "morgan2048_r2_T10" and bits_2048.data_dir != base.data_dir


def test_seed_epochs_and_energies_do_not_move_the_artifacts():
    base = layout()
    other = layout(
        experiment={"seed": 7},
        training={"epochs": 99},
        architecture={"e_mac": 4.6, "e_ac": 0.9},
    )
    assert other.point_dir == base.point_dir and other.data_dir == base.data_dir


def test_an_explicit_fingerprint_selects_another_point_of_the_same_config():
    config = parse_config(RAW)
    lay = ResultsLayout("results", config, FingerprintConfig(type="morgan", n_bits=2048, radius=2))
    assert lay.point_id == "morgan2048_r2_T10"
    assert ResultsLayout("results", config).point_id == "morgan1024_r2_T10"


def test_results_dir_may_be_a_string_or_a_path(tmp_path):
    assert layout(str(tmp_path)).point_dir == layout(tmp_path).point_dir


@pytest.mark.parametrize("method", ["model_path", "metrics_path", "curve_path", "opreport_path"])
def test_unknown_model_kinds_are_rejected(method):
    with pytest.raises(ValueError, match="model kind"):
        getattr(layout(), method)("cnn")


def test_kinds():
    assert KINDS == ("ann", "snn")
