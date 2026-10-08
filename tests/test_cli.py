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
