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

    @property
    def group(self) -> str:
        """Fingerprint family used to group points that differ only in T, e.g. `morgan1024`."""
        return f"{self.fingerprint}{self.n_bits}"
