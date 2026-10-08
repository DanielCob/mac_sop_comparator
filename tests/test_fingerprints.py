from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem, DataStructs
from rdkit.Chem import MACCSkeys, rdFingerprintGenerator

from mac_sop_comparator.config import FingerprintConfig, load_config
from mac_sop_comparator.data import FingerprintGenerator, InvalidSmilesError

CONTROL = ["CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O", "CCN1C(=O)NC(c2ccccc2)C1=O"]


@pytest.mark.parametrize("smiles", CONTROL)
def test_maccs_matches_rdkit_without_bit_zero(smiles):
    expected = np.array(list(MACCSkeys.GenMACCSKeys(Chem.MolFromSmiles(smiles))))[1:]
    fp = FingerprintGenerator("maccs").generate(smiles)
    assert fp.shape == (166,) and fp.dtype == np.float32
    assert np.array_equal(fp, expected)


@pytest.mark.parametrize("n_bits", [1024, 2048])
@pytest.mark.parametrize("smiles", CONTROL)
def test_morgan_matches_rdkit_bit_vector(smiles, n_bits):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=n_bits)
    expected = np.zeros(n_bits, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(gen.GetFingerprint(Chem.MolFromSmiles(smiles)), expected)
    fp = FingerprintGenerator("morgan", n_bits, 2).generate(smiles)
    assert fp.shape == (n_bits,)
    assert set(np.unique(fp)) <= {0.0, 1.0}
    assert np.array_equal(fp, expected)


def test_radius_changes_the_fingerprint():
    a = FingerprintGenerator("morgan", 1024, 1).generate("CC(=O)Oc1ccccc1C(=O)O")
    b = FingerprintGenerator("morgan", 1024, 3).generate("CC(=O)Oc1ccccc1C(=O)O")
    assert not np.array_equal(a, b)


def test_n_bits_per_type_and_from_config():
    assert FingerprintGenerator("maccs", n_bits=1024).n_bits == 166  # n_bits is ignored for maccs
    cfg = FingerprintConfig(type="morgan", n_bits=2048, radius=2)
    assert FingerprintGenerator.from_config(cfg).n_bits == 2048
    with pytest.raises(ValueError, match="type"):
        FingerprintGenerator("ecfp")


@pytest.mark.parametrize("bad", ["not a smiles", "", "C(C", None])
def test_invalid_smiles_raise(bad):
    with pytest.raises(InvalidSmilesError):
        FingerprintGenerator("maccs").generate(bad)


@pytest.mark.parametrize("kind", ["maccs", "morgan"])
def test_generate_many_skips_invalid_and_reports_positions(kind, caplog):
    smiles = ["CCO", "bad!", "c1ccccc1", "", "CCC"]
    X, kept = FingerprintGenerator(kind).generate_many(smiles)
    assert kept.tolist() == [0, 2, 4]
    assert X.shape == (3, FingerprintGenerator(kind).n_bits)
    assert np.array_equal(X[1], FingerprintGenerator(kind).generate("c1ccccc1"))
    assert "2 of 5" in caplog.text


def test_generate_many_all_invalid_gives_empty_matrix():
    X, kept = FingerprintGenerator("maccs").generate_many(["x!", ""])
    assert X.shape == (0, 166) and len(kept) == 0


def test_deterministic():
    g = FingerprintGenerator("morgan", 1024, 2)
    assert np.array_equal(g.generate("CCN1C(=O)NC(c2ccccc2)C1=O"), g.generate("CCN1C(=O)NC(c2ccccc2)C1=O"))


REAL = Path(__file__).parent.parent / "data_raw" / "tox21.csv.gz"


@pytest.mark.skipif(not REAL.is_file(), reason="tox21.csv.gz not cached in data_raw/")
@pytest.mark.parametrize(
    "gen, sparsity", [(("maccs",), 0.816), (("morgan", 1024, 2), 0.974), (("morgan", 2048, 2), 0.987)]
)
def test_measured_sparsity_on_tox21_sr_are(gen, sparsity):
    """Regression on the values measured for Tox21 / SR-ARE (D-34): they differ from Küppers 2024."""
    df = pd.read_csv(REAL).dropna(subset=["SR-ARE"])
    X, kept = FingerprintGenerator(*gen).generate_many(df["smiles"].tolist())
    assert len(df) - len(kept) == 7
    assert 1 - X.mean() == pytest.approx(sparsity, abs=0.005)


def test_2048_config_is_valid():
    cfg = load_config(Path(__file__).parent.parent / "configs" / "tox21_sr_are_2048.yaml")
    assert cfg.data.fingerprint.n_bits == 2048 and cfg.experiment.name == "tox21_sr_are_2048"
