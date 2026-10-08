from pathlib import Path

import numpy as np
import pytest

from mac_sop_comparator.config import SplitConfig
from mac_sop_comparator.data import Splitter


@pytest.fixture
def y():
    rng = np.random.default_rng(0)
    return (rng.random(1000) < 0.16).astype(int)  # ~16 % positive, like SR-ARE


def test_partitions_are_disjoint_and_cover_everything(y):
    parts = Splitter(0.8, 0.1, 0.1, seed=42).split(y)
    joined = np.concatenate(list(parts.values()))
    assert sorted(joined.tolist()) == list(range(len(y)))
    assert len(set(joined.tolist())) == len(y)


def test_sizes_follow_the_proportions(y):
    parts = Splitter(0.8, 0.1, 0.1, seed=42).split(y)
    assert (len(parts["train"]), len(parts["val"]), len(parts["test"])) == (800, 100, 100)


def test_stratification_keeps_the_positive_rate(y):
    parts = Splitter(0.8, 0.1, 0.1, seed=42).split(y)
    for idx in parts.values():
        assert y[idx].mean() == pytest.approx(y.mean(), abs=0.01)


def test_same_seed_is_stable_different_seed_differs(y):
    a = Splitter(0.8, 0.1, 0.1, seed=42).split(y)
    b = Splitter(0.8, 0.1, 0.1, seed=42).split(y)
    c = Splitter(0.8, 0.1, 0.1, seed=7).split(y)
    assert all(np.array_equal(a[k], b[k]) for k in a)
    assert not np.array_equal(a["test"], c["test"])


def test_indices_are_sorted(y):
    for idx in Splitter(0.7, 0.15, 0.15, seed=1).split(y).values():
        assert np.all(np.diff(idx) > 0)


def test_from_config():
    s = Splitter.from_config(SplitConfig(train=0.6, val=0.2, test=0.2), seed=3)
    assert (s.train, s.val, s.test, s.seed) == (0.6, 0.2, 0.2, 3)


def test_proportions_must_sum_to_one():
    with pytest.raises(ValueError, match="sum to 1"):
        Splitter(0.8, 0.1, 0.2, seed=0)


def test_too_few_samples_in_a_class_explains_itself():
    y = np.array([0] * 20 + [1])  # a single positive cannot be stratified
    with pytest.raises(ValueError, match="class counts"):
        Splitter(0.8, 0.1, 0.1, seed=0).split(y)


REAL = Path(__file__).parent.parent / "data_raw" / "tox21.csv.gz"


@pytest.mark.skipif(not REAL.is_file(), reason="tox21.csv.gz not cached in data_raw/")
def test_real_tox21_split_is_stratified_and_disjoint():
    from mac_sop_comparator.config import load_config
    from mac_sop_comparator.data import MoleculeNetLoader
    from mac_sop_comparator.data.cleaning import clean_molecules

    cfg = load_config(Path(__file__).parent.parent / "configs" / "tox21_sr_are.yaml")
    labels = clean_molecules(MoleculeNetLoader().load(cfg.data))[0]["label"].to_numpy()
    parts = Splitter.from_config(cfg.data.split, cfg.experiment.seed).split(labels)
    assert sum(len(v) for v in parts.values()) == len(labels) == 5825
    assert all(abs(labels[v].mean() - labels.mean()) < 0.005 for v in parts.values())
