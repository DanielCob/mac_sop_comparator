from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mac_sop_comparator.config import load_config
from mac_sop_comparator.data import FingerprintGenerator, MoleculeNetLoader
from mac_sop_comparator.data.cleaning import canonical_smiles, clean_molecules, signature_report


def frame(rows):
    return pd.DataFrame(rows, columns=["smiles", "label"])


def test_canonical_smiles_and_invalid():
    assert canonical_smiles("OCC") == canonical_smiles("CCO") == "CCO"
    assert canonical_smiles("not a smiles") is None
    assert canonical_smiles("") is None
    assert canonical_smiles(None) is None


def test_duplicates_with_same_label_keep_one():
    df = frame([("CCO", 1), ("CCC", 0), ("OCC", 1), ("CCO", 1)])  # OCC is CCO written differently
    out, rep = clean_molecules(df)
    assert out["smiles"].tolist() == ["CCO", "CCC"]  # first-appearance order
    assert out["label"].tolist() == [1, 0]
    assert (rep.n_input, rep.n_duplicates_removed, rep.n_conflicting_molecules, rep.n_output) == (4, 2, 0, 2)


def test_conflicting_labels_remove_every_row():
    df = frame([("CCO", 1), ("CCC", 0), ("OCC", 0), ("c1ccccc1", 1)])
    out, rep = clean_molecules(df)
    assert out["smiles"].tolist() == ["CCC", "c1ccccc1"]
    assert (rep.n_conflicting_molecules, rep.n_conflicting_rows, rep.n_duplicates_removed) == (1, 2, 0)


def test_invalid_smiles_are_discarded_and_counted():
    out, rep = clean_molecules(frame([("CCO", 1), ("bad!", 0), ("", 1)]))
    assert out["smiles"].tolist() == ["CCO"]
    assert rep.n_invalid == 2 and rep.n_output == 1


def test_stereoisomers_are_distinct_molecules():
    out, rep = clean_molecules(frame([("C[C@H](N)C(=O)O", 1), ("C[C@@H](N)C(=O)O", 0)]))
    assert len(out) == 2 and rep.n_conflicting_molecules == 0


def test_output_is_clean_frame_and_report_adds_up():
    df = frame([("CCO", 1), ("OCC", 1), ("CCC", 0), ("CCC", 1), ("x!", 0), ("CCCC", 0)])
    out, rep = clean_molecules(df)
    assert list(out.columns) == ["smiles", "label"] and out.index.tolist() == list(range(len(out)))
    assert rep.n_input == rep.n_invalid + rep.n_duplicates_removed + rep.n_conflicting_rows + rep.n_output
    assert not out["smiles"].duplicated().any()
    assert rep.to_dict()["n_output"] == len(out)


def test_signature_report_counts_repeats_and_conflicts():
    X = np.array([[1, 0], [1, 0], [0, 1], [0, 1], [0, 1], [1, 1]], dtype=np.float32)
    y = np.array([1, 1, 0, 1, 0, 1])  # [1,0] repeated, consistent; [0,1] repeated, conflicting
    rep = signature_report(X, y)
    assert rep.n_samples == 6 and rep.n_unique == 3
    assert (rep.n_repeated_signatures, rep.n_samples_in_repeated) == (2, 5)
    assert (rep.n_conflicting_signatures, rep.n_samples_in_conflicting) == (1, 3)


def test_signature_report_without_repeats():
    rep = signature_report(np.eye(3, dtype=np.float32), np.array([0, 1, 0]))
    assert rep.n_repeated_signatures == 0 and rep.n_conflicting_signatures == 0


REAL = Path(__file__).parent.parent / "data_raw" / "tox21.csv.gz"


@pytest.mark.skipif(not REAL.is_file(), reason="tox21.csv.gz not cached in data_raw/")
def test_real_tox21_sr_are_pipeline():
    cfg = load_config(Path(__file__).parent.parent / "configs" / "tox21_sr_are.yaml")
    out, rep = clean_molecules(MoleculeNetLoader().load(cfg.data))
    assert (rep.n_input, rep.n_invalid, rep.n_duplicates_removed, rep.n_conflicting_molecules) == (5832, 7, 0, 0)
    X, kept = FingerprintGenerator("maccs").generate_many(out["smiles"].tolist())
    assert len(kept) == len(out)  # nothing invalid is left after cleaning
    rep_fp = signature_report(X, out["label"].to_numpy())
    assert rep_fp.n_repeated_signatures > 0 and rep_fp.n_conflicting_signatures > 0  # reported, not removed
