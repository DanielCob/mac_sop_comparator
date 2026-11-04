"""PointResult: everything measured at one design-space point, the input of the report builder."""

from dataclasses import dataclass

from mac_sop_comparator.comparison.metrics import MetricSet
from mac_sop_comparator.config import ArchitectureParams
from mac_sop_comparator.profiling import OpReport


@dataclass(frozen=True)
class PointResult:
    """One (fingerprint, n_bits, T) point: test accuracy of both models, their OpReports and the metrics."""

    point_id: str
    fingerprint: str
    n_bits: int
    T: int
    ann_auc: float
    snn_auc: float
    ann_balanced_accuracy: float
    snn_balanced_accuracy: float
    ann_report: OpReport
    snn_report: OpReport
    metrics: MetricSet
    params: ArchitectureParams
    hidden_layers: tuple[int, ...] = ()  # empty when unknown
    seed: int | None = None

    @property
    def group(self) -> str:
        """Family of points that differ only in T, e.g. `morgan1024`, plus topology and seed when known."""
        hidden = f"_h{'-'.join(map(str, self.hidden_layers))}" if self.hidden_layers else ""
        seed = f"_s{self.seed}" if self.seed is not None else ""
        return f"{self.fingerprint}{self.n_bits}{hidden}{seed}"
