"""End-to-end flow on a small synthetic dataset (200 molecules, 2 epochs, CPU): the whole tool through its CLI."""

import csv
import json

import pytest
import yaml
from conftest import CONFIG

from mac_sop_comparator.experiments.cli import main
from mac_sop_comparator.profiling import OpReport

POINT = "morgan1024_r2_T4"


def write_config(workdir, name="cpu.yaml", fingerprint=None):
    raw = {
        **CONFIG,
        "experiment": {**CONFIG["experiment"], "device": "cpu"},
        "training": {**CONFIG["training"], "epochs": 2},
    }
    if fingerprint:
        raw["data"] = {**CONFIG["data"], "fingerprint": fingerprint}
    (workdir / name).write_text(yaml.safe_dump(raw))
    return name


def cli(*args):
    assert main(list(args)) == 0


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


@pytest.fixture
def config_file(workdir):
    return write_config(workdir)


def test_run_leaves_a_complete_and_consistent_set_of_artifacts(workdir, config_file):
    cli("run", "--config", config_file, "--results-dir", "out")
    root = workdir / "out" / "demo"
    point = root / POINT
    assert (root / "data" / "morgan1024_r2" / "dataset.npz").is_file() and (root / "run.log").is_file()
    for kind in ("ann", "snn"):
        assert all((point / name).is_file() for name in (f"{kind}.pt", f"{kind}_metrics.json", f"{kind}_curve.png", f"opreport_{kind}.json"))

    (row,) = read_csv(point / "report" / "metrics.csv")
    ann_metrics = json.loads((point / "ann_metrics.json").read_text())
    snn_metrics = json.loads((point / "snn_metrics.json").read_text())
    ann, snn = OpReport.load(point / "opreport_ann.json"), OpReport.load(point / "opreport_snn.json")
    assert row["point_id"] == POINT and int(row["T"]) == 4 and int(row["n_bits"]) == 1024
    assert float(row["ann_auc"]) == pytest.approx(ann_metrics["test"]["roc_auc"])
    assert float(row["snn_auc"]) == pytest.approx(snn_metrics["test"]["roc_auc"])
    assert float(row["n_mac"]) == ann.macs == 1024 * 8 + 8 * 2
    assert float(row["n_sop"]) == pytest.approx(snn.sops) and float(row["n_upd"]) == snn.state_updates == 4 * (8 + 2)
    assert float(row["r"]) == pytest.approx(snn.sops / (4 * ann.macs))
    assert float(row["cost_snn"]) == pytest.approx(snn.sops * 0.2 + snn.state_updates * 0.3)
    assert float(row["cost_ratio"]) == pytest.approx(float(row["cost_snn"]) / float(row["cost_ann"]))
    assert float(row["r_critical"]) == pytest.approx((ann.macs - 0.3 * snn.state_updates) / (4 * 0.2 * ann.macs))
    assert (row["snn_cheaper"] == "True") == (float(row["r"]) < float(row["r_critical"]))

    data = json.loads((point / "report" / "opreports.json").read_text())
    assert len(data) == 1 and len(data[0]["metrics"]["rate_per_layer"]) == 3  # input, hidden, output
    for name in ("operations", "auc_vs_cost", "rates"):
        assert (point / "report" / f"{name}.png").read_bytes().startswith(b"\x89PNG")


def test_run_goes_through_every_state_in_the_log(workdir, config_file):
    cli("run", "--config", config_file, "--results-dir", "out")
    log = (workdir / "out" / "demo" / "run.log").read_text()
    states = ["idle", "configured", "data_prepared", "training", "evaluating", "measuring", "comparing", "reporting", "idle"]
    positions = [log.index(f"state: {a} -> {b}") for a, b in zip(states, states[1:])]
    assert positions == sorted(positions)


def test_run_equals_the_separate_commands(workdir, config_file):
    cli("run", "--config", config_file, "--results-dir", "joint")
    for command in (["prepare"], ["train", "--model", "ann"], ["train", "--model", "snn"], ["profile"], ["compare"]):
        cli(*command, "--config", config_file, "--results-dir", "split")
    for name in ("metrics.csv", "opreports.json"):
        joint = (workdir / "joint" / "demo" / POINT / "report" / name).read_text()
        split = (workdir / "split" / "demo" / POINT / "report" / name).read_text()
        assert joint == split, name


def test_a_second_run_reproduces_the_first(workdir, config_file):
    cli("run", "--config", config_file, "--results-dir", "first")
    cli("run", "--config", config_file, "--results-dir", "second")
    first = (workdir / "first" / "demo" / POINT / "report" / "metrics.csv").read_text()
    assert first == (workdir / "second" / "demo" / POINT / "report" / "metrics.csv").read_text()


def test_maccs_point_uses_166_inputs(workdir):
    name = write_config(workdir, "maccs.yaml", {"type": "maccs", "n_bits": 1024, "radius": 2})
    cli("run", "--config", name, "--results-dir", "out")
    (row,) = read_csv(workdir / "out" / "demo" / "maccs_T4" / "report" / "metrics.csv")
    assert row["fingerprint"] == "maccs" and int(row["n_bits"]) == 166
    assert float(row["n_mac"]) == 166 * 8 + 8 * 2


def test_compare_refreshes_the_reports_from_the_saved_artifacts(workdir, config_file):
    cli("run", "--config", config_file, "--results-dir", "out")
    report = workdir / "out" / "demo" / POINT / "report"
    original = (report / "metrics.csv").read_text()
    (report / "metrics.csv").unlink()
    cli("compare", "--config", config_file, "--results-dir", "out")
    assert (report / "metrics.csv").read_text() == original
