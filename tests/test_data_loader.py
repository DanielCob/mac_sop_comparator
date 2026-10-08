import gzip
import hashlib
from pathlib import Path

import pytest

from mac_sop_comparator.config import DataConfig
from mac_sop_comparator.data import DataError, DatasetSource, MoleculeNetLoader

CSV = "TASK-A,TASK-B,mol_id,smiles\n1.0,0.0,m1,CCO\n0.0,,m2,CC\n,1.0,m3,CCC\n1.0,1.0,m4,c1ccccc1\n"


def data_cfg(task="TASK-A", dataset="fake") -> DataConfig:
    return DataConfig.model_validate(
        {
            "dataset": dataset,
            "task": task,
            "fingerprint": {"type": "morgan", "n_bits": 1024, "radius": 2},
            "split": {"train": 0.8, "val": 0.1, "test": 0.1},
        }
    )


@pytest.fixture
def cache(tmp_path):
    """A cache dir already holding a fake dataset, plus its source (no network involved)."""
    payload = gzip.compress(CSV.encode())
    (tmp_path / "fake.csv.gz").write_bytes(payload)
    source = DatasetSource(url="https://example.invalid/fake.csv.gz", sha256=hashlib.sha256(payload).hexdigest())
    return MoleculeNetLoader(tmp_path, {"fake": source})


def test_missing_labels_are_dropped(cache):
    df = cache.load(data_cfg("TASK-A"))
    assert list(df.columns) == ["smiles", "label"]
    assert df["smiles"].tolist() == ["CCO", "CC", "c1ccccc1"]
    assert df["label"].tolist() == [1, 0, 1]
    assert df["label"].dtype.kind == "i"


def test_task_selects_its_own_column(cache):
    assert cache.load(data_cfg("TASK-B"))["smiles"].tolist() == ["CCO", "CCC", "c1ccccc1"]


def test_unknown_task_lists_available(cache):
    with pytest.raises(DataError, match="TASK-Z.*TASK-A"):
        cache.load(data_cfg("TASK-Z"))


def test_unknown_dataset(cache):
    with pytest.raises(DataError, match="data.dataset 'nope'"):
        cache.load(data_cfg(dataset="nope"))


def test_hash_mismatch_is_rejected(cache, tmp_path):
    (tmp_path / "fake.csv.gz").write_bytes(gzip.compress(b"tampered"))
    with pytest.raises(DataError, match="SHA-256"):
        cache.load(data_cfg())


def test_download_failure_names_the_file(tmp_path):
    source = DatasetSource(url="file:///nonexistent/fake.csv.gz", sha256="0" * 64)
    with pytest.raises(DataError, match="save it manually"):
        MoleculeNetLoader(tmp_path, {"fake": source}).load(data_cfg())
    assert not list(tmp_path.glob("*.part"))


REAL = Path(__file__).parent.parent / "data_raw" / "tox21.csv.gz"


@pytest.mark.skipif(not REAL.is_file(), reason="tox21.csv.gz not cached in data_raw/")
def test_real_tox21_sr_are():
    df = MoleculeNetLoader(REAL.parent).load(data_cfg("SR-ARE", "tox21"))
    assert len(df) == 5832
    assert int(df["label"].sum()) == 942
    assert set(df["label"]) == {0, 1}
