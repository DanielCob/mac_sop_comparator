import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from mac_sop_comparator.config import FingerprintConfig, parse_config
from mac_sop_comparator.data import (
    DatasetSource,
    MoleculeNetLoader,
    fingerprint_id,
    load_prepared,
    prepare_data,
    prepared_dir,
    save_prepared,
)

N = 200  # alkane chains C, CC, CCC, ... : all valid and distinct


@pytest.fixture
def loader(tmp_path):
    rows = ["TASK,smiles"] + [f"{1 if i % 4 == 0 else 0},{'C' * (i + 1)}" for i in range(N)]
    rows.append(",CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC")  # unlabeled
    payload = gzip.compress("\n".join(rows).encode())
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "fake.csv.gz").write_bytes(payload)
    src = DatasetSource("https://example.invalid/fake.csv.gz", hashlib.sha256(payload).hexdigest())
    return MoleculeNetLoader(cache, {"fake": src})


@pytest.fixture
def config():
    return parse_config(
        {
            "experiment": {"name": "demo", "seed": 42},
            "data": {
                "dataset": "fake",
                "task": "TASK",
                "fingerprint": {"type": "morgan", "n_bits": 1024, "radius": 2},
                "split": {"train": 0.8, "val": 0.1, "test": 0.1},
            },
            "model": {"hidden_layers": [8], "n_outputs": 2, "snn": {"T": 4, "beta": 0.9, "threshold": 1.0}},
            "training": {"optimizer": "adam", "lr": 1e-3, "batch_size": 16, "epochs": 1},
            "architecture": {"weight_bits": 32, "e_mac": 1, "e_ac": 0.2, "e_upd": 0.3, "memory_scenarios": ["von_neumann"]},
        }
    )


def test_prepare_data(config, loader):
    data = prepare_data(config, loader=loader)
    assert data.X.shape == (N, 1024) and data.y.shape == (N,) and len(data.smiles) == N
    sizes = {k: len(v) for k, v in data.indices.items()}
    assert sizes == {"train": 160, "val": 20, "test": 20}
    s = data.summary
    assert s["n_molecules"] == N and s["n_positive"] == 50 and s["positive_rate"] == 0.25
    assert s["fingerprint"] == "morgan1024_r2" and s["n_features"] == 1024
    assert s["cleaning"]["n_input"] == N  # the unlabeled row never reaches cleaning
    assert s["partitions"]["test"]["positive_rate"] == pytest.approx(0.25, abs=0.06)
    assert set(data.partitions()) == {"train", "val", "test"} and len(data.partition("val")) == 20


def test_prepare_is_deterministic_and_seed_dependent(config, loader):
    a, b = prepare_data(config, loader=loader), prepare_data(config, loader=loader)
    assert all(np.array_equal(a.indices[k], b.indices[k]) for k in a.indices)
    other = config.model_copy(update={"experiment": config.experiment.model_copy(update={"seed": 7})})
    assert not np.array_equal(prepare_data(other, loader=loader).indices["test"], a.indices["test"])


def test_fingerprint_override_for_sweeps(config, loader):
    maccs = prepare_data(config, FingerprintConfig(type="maccs", n_bits=1024, radius=2), loader)
    assert maccs.X.shape[1] == 166 and maccs.summary["fingerprint"] == "maccs"


def test_save_and_load_roundtrip(config, loader, tmp_path):
    data = prepare_data(config, loader=loader)
    out = save_prepared(data, prepared_dir(tmp_path / "results", config))
    assert out == tmp_path / "results" / "demo" / "data" / "morgan1024_r2"
    assert (out / "dataset.npz").is_file() and json.loads((out / "summary.json").read_text()) == data.summary
    back = load_prepared(out)
    assert back.X.dtype == np.float32 and back.y.dtype == np.int64
    assert np.array_equal(back.X, data.X) and np.array_equal(back.y, data.y)
    assert back.smiles == data.smiles and back.summary == data.summary
    assert all(np.array_equal(back.indices[k], data.indices[k]) for k in data.indices)


def test_directories_differ_per_fingerprint(config, tmp_path):
    ids = {
        fingerprint_id(FingerprintConfig(type="maccs", n_bits=1024, radius=2)),
        fingerprint_id(FingerprintConfig(type="morgan", n_bits=1024, radius=2)),
        fingerprint_id(FingerprintConfig(type="morgan", n_bits=2048, radius=2)),
    }
    assert ids == {"maccs", "morgan1024_r2", "morgan2048_r2"}
    assert prepared_dir(tmp_path, config, FingerprintConfig(type="maccs", n_bits=9, radius=9)).name == "maccs"


def test_load_prepared_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="prepare"):
        load_prepared(tmp_path / "nothing")


REAL = Path(__file__).parent.parent / "data_raw" / "tox21.csv.gz"


@pytest.mark.skipif(not REAL.is_file(), reason="tox21.csv.gz not cached in data_raw/")
def test_real_tox21_pipeline_and_roundtrip(tmp_path):
    from mac_sop_comparator.config import load_config

    cfg = load_config(Path(__file__).parent.parent / "configs" / "tox21_sr_are.yaml")
    data = prepare_data(cfg)
    assert data.X.shape == (5825, 1024) and data.summary["n_positive"] == 942
    back = load_prepared(save_prepared(data, tmp_path))
    assert np.array_equal(back.X, data.X)
