"""Fixtures shared by the CLI and end-to-end tests: a fake `tox21` dataset cached in a temporary working directory."""

import gzip
import hashlib
import logging

import pytest
import yaml

from mac_sop_comparator.data import DatasetSource, loader

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
    "architecture": {"weight_bits": 32, "e_mac": 1, "e_ac": 0.2, "e_upd": 0.3, "e_byte_von_neumann": 53, "e_byte_neuromorphic": 1.3},
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
