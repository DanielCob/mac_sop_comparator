"""SMILES -> binary molecular fingerprints (MACCS and Morgan) with RDKit."""

import logging

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import MACCSkeys, rdFingerprintGenerator

from mac_sop_comparator.config import FingerprintConfig

logger = logging.getLogger(__name__)

MACCS_BITS = 166  # RDKit returns 167 positions; bit 0 is unused and dropped (D-08)


class InvalidSmilesError(ValueError):
    """RDKit could not parse the SMILES string."""


class FingerprintGenerator:
    def __init__(self, type: str, n_bits: int = 1024, radius: int = 2):
        if type not in ("maccs", "morgan"):
            raise ValueError(f"fingerprint type must be 'maccs' or 'morgan', got '{type}'")
        self.type = type
        self.n_bits = MACCS_BITS if type == "maccs" else n_bits
        self.radius = radius
        self._morgan = (
            rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits) if type == "morgan" else None
        )

    @classmethod
    def from_config(cls, config: FingerprintConfig) -> "FingerprintGenerator":
        return cls(config.type, config.n_bits, config.radius)

    def generate(self, smiles: str) -> np.ndarray:
        """Return the fingerprint as a float32 vector of zeros and ones, shape (n_bits,)."""
        mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
        if mol is None or mol.GetNumAtoms() == 0:
            raise InvalidSmilesError(f"invalid SMILES: {smiles!r}")
        if self._morgan is not None:
            return self._morgan.GetFingerprintAsNumPy(mol).astype(np.float32)
        bits = np.zeros(167, dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(MACCSkeys.GenMACCSKeys(mol), bits)
        return bits[1:].astype(np.float32)

    def generate_many(self, smiles: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Fingerprint a list of SMILES, skipping the invalid ones.

        Returns (X, kept) where X has shape (n_valid, n_bits) and `kept` holds the positions
        in `smiles` of the molecules that were converted.
        """
        rows, kept = [], []
        RDLogger.DisableLog("rdApp.*")  # RDKit prints every parse failure; we count them instead
        try:
            for i, s in enumerate(smiles):
                try:
                    rows.append(self.generate(s))
                    kept.append(i)
                except InvalidSmilesError:
                    pass
        finally:
            RDLogger.EnableLog("rdApp.*")
        if len(kept) < len(smiles):
            logger.warning("%d of %d SMILES were invalid and discarded", len(smiles) - len(kept), len(smiles))
        X = np.stack(rows) if rows else np.zeros((0, self.n_bits), dtype=np.float32)
        return X, np.asarray(kept, dtype=np.int64)
