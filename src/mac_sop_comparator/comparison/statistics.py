"""Seed statistics: mean and standard deviation of every metric over the seeds of each configuration (D-27, D-44)."""

import csv
import statistics
from collections.abc import Callable
from pathlib import Path

from mac_sop_comparator.comparison.point import PointResult

METRICS: dict[str, Callable[[PointResult], float]] = {
    "ann_auc": lambda p: p.ann_auc,
    "snn_auc": lambda p: p.snn_auc,
    "auc_gap": lambda p: p.ann_auc - p.snn_auc,  # computed per seed, then aggregated
    "ann_balanced_accuracy": lambda p: p.ann_balanced_accuracy,
    "snn_balanced_accuracy": lambda p: p.snn_balanced_accuracy,
    "n_sop": lambda p: p.metrics.n_sop,
    "r": lambda p: p.metrics.r,
    "r_critical": lambda p: p.metrics.r_critical,
    "cost_ratio": lambda p: p.metrics.cost_ratio,
    "energy_ratio": lambda p: p.metrics.energy_ratio,
}
COLUMNS = ["config_id", "n_seeds", "seeds"] + [f"{name}_{stat}" for name in METRICS for stat in ("mean", "std")]


class SeedStatistics:
    """Groups the points that differ only in the seed. The std is the sample one (n - 1); with one seed it is blank."""

    @staticmethod
    def aggregate(results: list[PointResult]) -> list[dict]:
        if not results:
            raise ValueError("there are no results to aggregate")
        groups: dict[str, list[PointResult]] = {}
        for p in results:
            groups.setdefault(p.config_id, []).append(p)
        rows = []
        for config_id, points in groups.items():
            row = {"config_id": config_id, "n_seeds": len(points), "seeds": ";".join(str(p.seed) for p in points)}
            for name, getter in METRICS.items():
                values = [getter(p) for p in points]
                row[f"{name}_mean"] = statistics.fmean(values)
                row[f"{name}_std"] = statistics.stdev(values) if len(values) > 1 else ""
            rows.append(row)
        return rows

    @staticmethod
    def has_repeats(rows: list[dict]) -> bool:
        return any(row["n_seeds"] > 1 for row in rows)

    @classmethod
    def write_csv(cls, results: list[PointResult], path: str | Path) -> Path:
        rows = cls.aggregate(results)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        return path
