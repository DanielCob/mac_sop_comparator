"""Data preparation pipeline (CU-01) and its persistence.

loader -> cleaning -> fingerprints -> splitter -> MolecularDataset, saved per fingerprint under
`<results_dir>/<experiment name>/data/<fingerprint id>/{dataset.npz, summary.json}`.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mac_sop_comparator.config import Config, FingerprintConfig
from mac_sop_comparator.data.cleaning import clean_molecules, signature_report
from mac_sop_comparator.data.dataset import MolecularDataset
from mac_sop_comparator.data.fingerprints import FingerprintGenerator
from mac_sop_comparator.data.loader import MoleculeNetLoader
from mac_sop_comparator.data.splitter import PARTITIONS, Splitter

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreparedData:
    smiles: list[str]  # canonical SMILES, one per row of the full dataset
    X: np.ndarray  # (n, F) float32
    y: np.ndarray  # (n,) int64
    indices: dict[str, np.ndarray]  # "train" | "val" | "test" -> row indices
    summary: dict

    def partition(self, name: str) -> MolecularDataset:
        return MolecularDataset(self.X[self.indices[name]], self.y[self.indices[name]])

    def partitions(self) -> dict[str, MolecularDataset]:
        return {name: self.partition(name) for name in PARTITIONS}


def fingerprint_id(fp: FingerprintConfig) -> str:
    return "maccs" if fp.type == "maccs" else f"morgan{fp.n_bits}_r{fp.radius}"


def prepared_dir(results_dir: str | Path, config: Config, fp: FingerprintConfig | None = None) -> Path:
    fp = fp or config.data.fingerprint
    return Path(results_dir) / config.experiment.name / "data" / fingerprint_id(fp)


def prepare_data(
    config: Config, fingerprint: FingerprintConfig | None = None, loader: MoleculeNetLoader | None = None
) -> PreparedData:
    """Run the whole preparation. `fingerprint` overrides `config.data.fingerprint` (sweeps)."""
    fp = fingerprint or config.data.fingerprint
    raw = (loader or MoleculeNetLoader()).load(config.data)
    cleaned, cleaning = clean_molecules(raw)

    X, kept = FingerprintGenerator.from_config(fp).generate_many(cleaned["smiles"].tolist())
    cleaned = cleaned.iloc[kept].reset_index(drop=True)
    y = cleaned["label"].to_numpy(dtype=np.int64)

    indices = Splitter.from_config(config.data.split, config.experiment.seed).split(y)
    ds = MolecularDataset(X, y)
    summary = {
        "experiment": config.experiment.name,
        "dataset": config.data.dataset,
        "task": config.data.task,
        "fingerprint": fingerprint_id(fp),
        "seed": config.experiment.seed,
        "n_features": ds.n_features,
        "n_molecules": len(ds),
        "n_positive": int(y.sum()),
        "positive_rate": float(y.mean()),
        "sparsity": ds.sparsity(),
        "cleaning": cleaning.to_dict(),
        "signatures": signature_report(X, y).to_dict(),
        "partitions": {
            name: {
                "n": len(idx),
                "n_positive": int(y[idx].sum()),
                "positive_rate": float(y[idx].mean()),
                "sparsity": ds.subset(idx).sparsity(),
            }
            for name, idx in indices.items()
        },
    }
    logger.info(
        "prepared %s/%s [%s]: %d molecules, %d features, %.1f%% positive, sparsity %.3f",
        summary["dataset"], summary["task"], summary["fingerprint"], len(ds), ds.n_features,
        100 * summary["positive_rate"], summary["sparsity"],
    )
    return PreparedData(cleaned["smiles"].tolist(), X, y, indices, summary)


def save_prepared(data: PreparedData, directory: str | Path) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        directory / "dataset.npz",
        X=data.X.astype(np.uint8),  # fingerprints are 0/1; float32 is restored on load
        y=data.y,
        smiles=np.array(data.smiles),
        **{f"{name}_idx": idx for name, idx in data.indices.items()},
    )
    (directory / "summary.json").write_text(json.dumps(data.summary, indent=2) + "\n", encoding="utf-8")
    logger.info("saved prepared data to %s", directory)
    return directory


def load_prepared(directory: str | Path) -> PreparedData:
    directory = Path(directory)
    npz_path, summary_path = directory / "dataset.npz", directory / "summary.json"
    if not npz_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError(f"no prepared data in {directory}; run the 'prepare' command first")
    with np.load(npz_path) as f:
        return PreparedData(
            smiles=f["smiles"].tolist(),
            X=f["X"].astype(np.float32),
            y=f["y"].astype(np.int64),
            indices={name: f[f"{name}_idx"] for name in PARTITIONS},
            summary=json.loads(summary_path.read_text(encoding="utf-8")),
        )
