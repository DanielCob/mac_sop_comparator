import gzip
import hashlib
import json
import logging

import pytest
import yaml

from mac_sop_comparator.data import DatasetSource, loader
from mac_sop_comparator.experiments.cli import main

CSV = "TASK,smiles\n" + "".join(f"{1 if i % 4 == 0 else 0},{'C' * (i + 1)}\n" for i in range(200))

CONFIG = {
    "experiment": {"name": "demo", "seed": 42, "n_threads": 2},
    "data": {
        "dataset": "tox21",
        "task": "TASK",
        "fingerprint": {"type": "morgan", "n_bits": 1024, "radius": 2},
        "split": {"train": 0.8, "val": 0.1, "test": 0.1},
    },
    "model": {"hidden_layers": [8], "n_outputs": 2, "snn": {"T": 4, "beta": 0.9, "threshold": 1.0}},
    "training": {"optimizer": "adam", "lr": 1e-3, "batch_size": 16, "epochs": 1},
    "architecture": {"weight_bits": 32, "e_mac": 1, "e_ac": 0.2, "e_upd": 0.3, "memory_scenarios": ["von_neumann"]},
}


@pytest.fixture(autouse=True)
def reset_logging():
    yield
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    """A cwd with a fake `tox21` dataset cached in data_raw/ (no network) and a config file."""
    payload = gzip.compress(CSV.encode())
    (tmp_path / "data_raw").mkdir()
    (tmp_path / "data_raw" / "tox21.csv.gz").write_bytes(payload)
    fake = DatasetSource("https://example.invalid/tox21.csv.gz", hashlib.sha256(payload).hexdigest())
    monkeypatch.setitem(loader.SOURCES, "tox21", fake)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(CONFIG))
    return tmp_path


def test_prepare_writes_artifacts_and_log(workdir, capsys):
    assert main(["prepare", "--config", "c.yaml", "--results-dir", "out"]) == 0
    data_dir = workdir / "out" / "demo" / "data" / "morgan1024_r2"
    assert (data_dir / "dataset.npz").is_file()
    assert json.loads((data_dir / "summary.json").read_text())["n_molecules"] == 200
    assert "prepared 200 molecules" in capsys.readouterr().out
    log = (workdir / "out" / "demo" / "run.log").read_text()
    assert "experiment 'demo': seed 42, 2 CPU threads" in log and "torch " in log


def test_results_dir_defaults_to_results(workdir):
    assert main(["prepare", "--config", "c.yaml"]) == 0
    assert (workdir / "results" / "demo" / "data" / "morgan1024_r2" / "dataset.npz").is_file()


def test_missing_config_file(workdir, capsys):
    assert main(["prepare", "--config", "nope.yaml"]) == 2
    assert "not found" in capsys.readouterr().err


def test_invalid_config_names_the_field(workdir, capsys):
    bad = {**CONFIG, "model": {**CONFIG["model"], "snn": {"T": 0, "beta": 0.9, "threshold": 1.0}}}
    (workdir / "bad.yaml").write_text(yaml.safe_dump(bad))
    assert main(["prepare", "--config", "bad.yaml"]) == 2
    assert "model.snn.T" in capsys.readouterr().err


def test_dataset_not_available_is_a_clean_error(workdir, capsys):
    (workdir / "data_raw" / "tox21.csv.gz").unlink()  # nothing cached, and the URL does not resolve
    assert main(["prepare", "--config", "c.yaml"]) == 1
    assert "save it manually" in capsys.readouterr().err


def test_other_commands_are_not_implemented_yet(workdir, capsys):
    assert main(["compare", "--config", "c.yaml"]) == 1
    assert "not implemented" in capsys.readouterr().err


# --- train / profile (ANN) ---------------------------------------------------------------------


@pytest.fixture
def prepared(workdir):
    """workdir plus a CPU config (`cpu.yaml`) and the data already prepared into ./out."""
    cpu = {**CONFIG, "experiment": {**CONFIG["experiment"], "device": "cpu"}, "training": {**CONFIG["training"], "epochs": 2}}
    (workdir / "cpu.yaml").write_text(yaml.safe_dump(cpu))
    assert main(["prepare", "--config", "cpu.yaml", "--results-dir", "out"]) == 0
    return workdir


def test_train_ann_writes_checkpoint_metrics_and_opreport(prepared, capsys):
    assert main(["train", "--model", "ann", "--config", "cpu.yaml", "--results-dir", "out"]) == 0
    point = prepared / "out" / "demo" / "morgan1024_r2_T4"
    assert (point / "ann.pt").is_file() and (point / "ann_curve.png").stat().st_size > 0
    metrics = json.loads((point / "ann_metrics.json").read_text())
    assert metrics["topology"] == [1024, 8, 2] and metrics["device"] == "cpu" and metrics["epochs_run"] == 2
    assert set(metrics["validation"]) == set(metrics["test"]) == {"roc_auc", "balanced_accuracy"}
    assert len(metrics["history"]) == 2
    report = json.loads((point / "opreport_ann.json").read_text())
    assert report["architecture"] == "ann" and report["per_inference"]["macs"] == 1024 * 8 + 8 * 2
    out = capsys.readouterr().out
    assert "trained the ANN" in out and "8208 MACs" in out


def test_train_is_reproducible(prepared):
    def run():
        assert main(["train", "--model", "ann", "--config", "cpu.yaml", "--results-dir", "out"]) == 0
        return json.loads((prepared / "out" / "demo" / "morgan1024_r2_T4" / "ann_metrics.json").read_text())["history"]

    assert run() == run()


def test_profile_after_train(prepared, capsys):
    main(["train", "--model", "ann", "--config", "cpu.yaml", "--results-dir", "out"])
    capsys.readouterr()
    assert main(["profile", "--config", "cpu.yaml", "--results-dir", "out"]) == 0
    assert "ann: 8208 MACs, 10 biases" in capsys.readouterr().out


def test_train_without_prepared_data_says_what_to_run(workdir, capsys):
    assert main(["train", "--model", "ann", "--config", "c.yaml", "--results-dir", "out"]) == 1
    assert "prepare" in capsys.readouterr().err


def test_profile_without_a_trained_model_says_what_to_run(prepared, capsys):
    assert main(["profile", "--config", "cpu.yaml", "--results-dir", "out"]) == 1
    assert "'train' command" in capsys.readouterr().err


def test_train_snn_writes_its_artifacts(prepared, capsys):
    assert main(["train", "--model", "snn", "--config", "cpu.yaml", "--results-dir", "out"]) == 0
    point = prepared / "out" / "demo" / "morgan1024_r2_T4"
    assert (point / "snn.pt").is_file() and (point / "snn_curve.png").stat().st_size > 0
    metrics = json.loads((point / "snn_metrics.json").read_text())
    assert metrics["model"] == "snn" and metrics["snn"] == {"T": 4, "beta": 0.9, "threshold": 1.0, "reset": "subtract"}
    report = json.loads((point / "opreport_snn.json").read_text())
    assert report["architecture"] == "snn"
    per = report["per_inference"]
    assert per["macs"] is None and per["state_updates"] == 4 * (8 + 2) and len(per["spikes_per_layer"]) == 3
    assert "trained the SNN" in capsys.readouterr().out


def test_ann_and_snn_start_from_the_same_initial_weights(prepared):
    from mac_sop_comparator.models import load_checkpoint

    for kind in ("ann", "snn"):
        assert main(["train", "--model", kind, "--config", "cpu.yaml", "--results-dir", "out"]) == 0
    point = prepared / "out" / "demo" / "morgan1024_r2_T4"
    assert [h["epoch"] for h in json.loads((point / "snn_metrics.json").read_text())["history"]] == [1, 2]
    ann, snn = load_checkpoint(point / "ann.pt"), load_checkpoint(point / "snn.pt")
    assert [p.shape for p in ann.parameters()] == [p.shape for p in snn.parameters()]


def test_profile_reports_both_models_when_both_are_trained(prepared, capsys):
    for kind in ("ann", "snn"):
        main(["train", "--model", kind, "--config", "cpu.yaml", "--results-dir", "out"])
    capsys.readouterr()
    assert main(["profile", "--config", "cpu.yaml", "--results-dir", "out"]) == 0
    out = capsys.readouterr().out
    assert "ann: 8208 MACs" in out and "snn: " in out and "SOPs" in out
    assert "40 state updates" in out  # T * (8 + 2) = 4 * 10
