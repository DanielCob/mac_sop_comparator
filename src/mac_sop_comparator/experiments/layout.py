"""ResultsLayout: the one place that knows where every artifact of a point lives (D-25, D-40).

    results/<experiment>/
    ├── run.log
    ├── data/<fingerprint>/{dataset.npz, summary.json}
    └── <fingerprint>_T<T>/                 e.g. morgan1024_r2_T10
        ├── {ann,snn}.pt, {ann,snn}_metrics.json, {ann,snn}_curve.png, opreport_{ann,snn}.json
        └── report/{metrics.csv, opreports.json, *.png}

Separate commands find each other's artifacts by rebuilding the same layout from the same config.
"""

from pathlib import Path

from mac_sop_comparator.config import Config, FingerprintConfig
from mac_sop_comparator.data import fingerprint_id, prepared_dir

KINDS = ("ann", "snn")


class ResultsLayout:
    def __init__(self, results_dir: str | Path, config: Config, fp: FingerprintConfig | None = None):
        self.results_dir = Path(results_dir)
        self.config = config
        self.fp = fp or config.data.fingerprint

    @property
    def experiment_dir(self) -> Path:
        return self.results_dir / self.config.experiment.name

    @property
    def log_path(self) -> Path:
        return self.experiment_dir / "run.log"

    @property
    def data_dir(self) -> Path:
        """Prepared data; it does not depend on T, so every point of a fingerprint shares it."""
        return prepared_dir(self.results_dir, self.config, self.fp)

    @property
    def point_id(self) -> str:
        """Identifies a design-space point: fingerprint and SNN time steps, e.g. `morgan1024_r2_T10`."""
        return f"{fingerprint_id(self.fp)}_T{self.config.model.snn.T}"

    @property
    def point_dir(self) -> Path:
        return self.experiment_dir / self.point_id

    @property
    def report_dir(self) -> Path:
        return self.point_dir / "report"

    @staticmethod
    def _check(kind: str) -> str:
        if kind not in KINDS:
            raise ValueError(f"model kind must be one of {KINDS}, got '{kind}'")
        return kind

    def model_path(self, kind: str) -> Path:
        return self.point_dir / f"{self._check(kind)}.pt"

    def metrics_path(self, kind: str) -> Path:
        return self.point_dir / f"{self._check(kind)}_metrics.json"

    def curve_path(self, kind: str) -> Path:
        return self.point_dir / f"{self._check(kind)}_curve.png"

    def opreport_path(self, kind: str) -> Path:
        return self.point_dir / f"opreport_{self._check(kind)}.json"
