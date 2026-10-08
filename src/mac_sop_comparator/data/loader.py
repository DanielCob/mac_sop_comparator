"""Download, cache and read MoleculeNet datasets (D-07)."""

import hashlib
import logging
import shutil
import ssl
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import certifi
import pandas as pd

from mac_sop_comparator.config import DataConfig

logger = logging.getLogger(__name__)


class DataError(RuntimeError):
    """A dataset could not be obtained or does not have the expected content."""


@dataclass(frozen=True)
class DatasetSource:
    url: str
    sha256: str
    smiles_column: str = "smiles"

    @property
    def filename(self) -> str:
        return self.url.rsplit("/", 1)[-1]


SOURCES: dict[str, DatasetSource] = {
    "tox21": DatasetSource(
        url="https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz",
        sha256="45d09792492ce049039dd24aa27b07fc79ce20c573187d4d90bcd178c0c0d360",
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class MoleculeNetLoader:
    """Provides (smiles, label) for one task of a MoleculeNet dataset."""

    def __init__(self, cache_dir: str | Path = "data_raw", sources: dict[str, DatasetSource] | None = None):
        self.cache_dir = Path(cache_dir)
        self.sources = SOURCES if sources is None else sources

    def load(self, config: DataConfig) -> pd.DataFrame:
        """Return a DataFrame with columns `smiles` and `label` (int), without missing labels."""
        source = self._source(config.dataset)
        raw = pd.read_csv(self._ensure_file(config.dataset, source))
        if config.task not in raw.columns:
            tasks = [c for c in raw.columns if c != source.smiles_column]
            raise DataError(f"data.task '{config.task}' not in dataset '{config.dataset}'; available: {tasks}")
        df = raw[[source.smiles_column, config.task]].rename(
            columns={source.smiles_column: "smiles", config.task: "label"}
        )
        n_total = len(df)
        df = df.dropna(subset=["smiles", "label"]).reset_index(drop=True)
        df["label"] = df["label"].astype(int)
        logger.info(
            "%s/%s: %d molecules, %d without label dropped, %d kept (%d positive)",
            config.dataset, config.task, n_total, n_total - len(df), len(df), int(df["label"].sum()),
        )
        return df

    def _source(self, dataset: str) -> DatasetSource:
        if dataset not in self.sources:
            raise DataError(f"data.dataset '{dataset}' is not supported; available: {sorted(self.sources)}")
        return self.sources[dataset]

    def _ensure_file(self, dataset: str, source: DatasetSource) -> Path:
        path = self.cache_dir / source.filename
        if not path.is_file():
            self._download(source, path)
        if _sha256(path) != source.sha256:
            raise DataError(
                f"{path} does not match the expected SHA-256 for '{dataset}'; delete it to download again"
            )
        return path

    def _download(self, source: DatasetSource, path: Path) -> None:
        logger.info("Downloading %s", source.url)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".part")
        # certifi: the python.org build on macOS ships without root certificates
        context = ssl.create_default_context(cafile=certifi.where())
        try:
            with urllib.request.urlopen(source.url, context=context, timeout=60) as response, tmp.open("wb") as out:
                shutil.copyfileobj(response, out)
        except OSError as exc:
            tmp.unlink(missing_ok=True)
            raise DataError(f"could not download {source.url} ({exc}); save it manually as {path}") from exc
        tmp.replace(path)
