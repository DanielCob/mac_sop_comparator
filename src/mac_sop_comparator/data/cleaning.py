"""Dataset cleaning (D-10): duplicates by canonical SMILES; repeated fingerprints are only reported."""

import logging
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CleaningReport:
    n_input: int
    n_invalid: int  # SMILES RDKit cannot parse (discarded)
    n_duplicates_removed: int  # extra rows of molecules repeated with the same label (one is kept)
    n_conflicting_molecules: int  # distinct molecules repeated with different labels (all rows dropped)
    n_conflicting_rows: int  # rows dropped because of those conflicts
    n_output: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class SignatureReport:
    n_samples: int
    n_unique: int  # distinct fingerprints
    n_repeated_signatures: int  # fingerprints shared by 2+ molecules
    n_samples_in_repeated: int  # molecules whose fingerprint is shared
    n_conflicting_signatures: int  # shared fingerprints whose molecules have different labels
    n_samples_in_conflicting: int

    def to_dict(self) -> dict:
        return asdict(self)


def canonical_smiles(smiles: str) -> str | None:
    """Isomeric canonical SMILES, or None if RDKit cannot parse it (or it is empty)."""
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    return Chem.MolToSmiles(mol)


def clean_molecules(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """Canonicalize SMILES and remove duplicates.

    Input columns: `smiles`, `label`. Invalid SMILES are discarded. A molecule repeated with the
    same label is kept once; repeated with different labels it is removed entirely.
    Output: columns `smiles` (canonical) and `label`, index reset, first-appearance order.
    """
    RDLogger.DisableLog("rdApp.*")
    try:
        canon = df["smiles"].map(canonical_smiles)
    finally:
        RDLogger.EnableLog("rdApp.*")
    valid = canon.notna()
    work = pd.DataFrame({"smiles": canon[valid], "label": df.loc[valid, "label"]})

    n_labels = work.groupby("smiles", sort=False)["label"].transform("nunique")
    conflicting = n_labels > 1
    consistent = work[~conflicting]
    cleaned = consistent.drop_duplicates(subset="smiles", keep="first").reset_index(drop=True)

    report = CleaningReport(
        n_input=len(df),
        n_invalid=int((~valid).sum()),
        n_duplicates_removed=len(consistent) - len(cleaned),
        n_conflicting_molecules=int(work.loc[conflicting, "smiles"].nunique()),
        n_conflicting_rows=int(conflicting.sum()),
        n_output=len(cleaned),
    )
    logger.info("cleaning: %s", report)
    return cleaned, report


def signature_report(X: np.ndarray, y: np.ndarray) -> SignatureReport:
    """Count repeated fingerprints and those whose molecules disagree on the label (report only)."""
    _, inverse, counts = np.unique(X, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.reshape(-1)
    n_labels = np.array([len(set(y[inverse == g])) for g in np.flatnonzero(counts > 1)], dtype=int)
    repeated_ids = np.flatnonzero(counts > 1)
    conflicting_ids = repeated_ids[n_labels > 1]
    report = SignatureReport(
        n_samples=len(X),
        n_unique=len(counts),
        n_repeated_signatures=len(repeated_ids),
        n_samples_in_repeated=int(counts[repeated_ids].sum()),
        n_conflicting_signatures=len(conflicting_ids),
        n_samples_in_conflicting=int(counts[conflicting_ids].sum()),
    )
    logger.info("fingerprints: %s", report)
    return report
