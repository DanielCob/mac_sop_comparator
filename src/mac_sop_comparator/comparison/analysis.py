"""Statistical comparison of the ANN and the SNN over the seeds, and the tables of the report (D-50)."""

import csv
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

from scipy import stats

from mac_sop_comparator.comparison.point import PointResult

CONFIDENCE = 0.95
METRIC_GETTERS = {
    "roc_auc": (lambda p: p.ann_auc, lambda p: p.snn_auc),
    "balanced_accuracy": (lambda p: p.ann_balanced_accuracy, lambda p: p.snn_balanced_accuracy),
}


@dataclass(frozen=True)
class PairedComparison:
    """ANN against SNN on one configuration, paired by seed (a seed fixes the split and the initial weights)."""

    config_id: str
    metric: str
    n_seeds: int
    ann_mean: float
    snn_mean: float
    diff_mean: float  # ANN - SNN
    ci_low: float  # 95 % confidence interval of the mean difference (t distribution)
    ci_high: float
    p_value: float  # two-sided Wilcoxon signed-rank test


def paired_comparison(config_id: str, ann: list[float], snn: list[float], metric: str) -> PairedComparison:
    if len(ann) != len(snn) or len(ann) < 2:
        raise ValueError(f"a paired comparison needs two equal lists of at least 2 values, got {len(ann)} and {len(snn)}")
    diffs = [a - s for a, s in zip(ann, snn)]
    n, mean = len(diffs), statistics.fmean(diffs)
    half = stats.t.ppf(1 - (1 - CONFIDENCE) / 2, n - 1) * statistics.stdev(diffs) / math.sqrt(n)
    # with no difference at all the signed-rank statistic is undefined: the samples are identical, p = 1
    p = 1.0 if all(d == 0 for d in diffs) else float(stats.wilcoxon(ann, snn).pvalue)
    return PairedComparison(config_id, metric, n, statistics.fmean(ann), statistics.fmean(snn), mean, mean - half, mean + half, p)


class Analysis:
    @staticmethod
    def comparisons(results: list[PointResult], metric: str = "roc_auc") -> list[PairedComparison]:
        """One comparison per configuration with at least 2 seeds (the points that differ only in the seed)."""
        if metric not in METRIC_GETTERS:
            raise ValueError(f"metric must be one of {list(METRIC_GETTERS)}, got '{metric}'")
        ann_of, snn_of = METRIC_GETTERS[metric]
        groups: dict[str, list[PointResult]] = {}
        for p in results:
            groups.setdefault(p.config_id, []).append(p)
        return [
            paired_comparison(config_id, [ann_of(p) for p in pts], [snn_of(p) for p in pts], metric)
            for config_id, pts in groups.items()
            if len(pts) >= 2
        ]

    @classmethod
    def write_comparisons(cls, results: list[PointResult], path: str | Path) -> Path:
        rows = [asdict(c) for metric in METRIC_GETTERS for c in cls.comparisons(results, metric)]
        if not rows:
            raise ValueError("no configuration has more than one seed")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return path

    # Summary table of the report ----------------------------------------------------------------------

    @staticmethod
    def summary_rows(results: list[PointResult]) -> list[dict]:
        """One row per configuration: accuracy (mean, std), the gap with its test, and the efficiency figures."""
        if not results:
            raise ValueError("there are no results to summarize")
        groups: dict[str, list[PointResult]] = {}
        for p in results:
            groups.setdefault(p.config_id, []).append(p)
        rows = []
        for config_id, pts in groups.items():
            def mean(getter):
                return statistics.fmean(getter(p) for p in pts)

            def std(getter):
                return statistics.stdev([getter(p) for p in pts]) if len(pts) > 1 else None

            row = {"config_id": config_id, "n_seeds": len(pts)}
            for key, getter in (("ann_auc", lambda p: p.ann_auc), ("snn_auc", lambda p: p.snn_auc)):
                row[f"{key}_mean"], row[f"{key}_std"] = mean(getter), std(getter)
            if len(pts) > 1:
                c = paired_comparison(config_id, [p.ann_auc for p in pts], [p.snn_auc for p in pts], "roc_auc")
                row.update(auc_gap=c.diff_mean, auc_gap_ci_low=c.ci_low, auc_gap_ci_high=c.ci_high, auc_gap_p=c.p_value)
            else:
                row.update(auc_gap=pts[0].ann_auc - pts[0].snn_auc, auc_gap_ci_low=None, auc_gap_ci_high=None, auc_gap_p=None)
            for key, getter in (
                ("n_mac", lambda p: p.metrics.n_mac), ("n_sop", lambda p: p.metrics.n_sop), ("n_upd", lambda p: p.metrics.n_upd),
                ("r", lambda p: p.metrics.r), ("r_critical", lambda p: p.metrics.r_critical),
                ("cost_ratio", lambda p: p.metrics.cost_ratio), ("bytes_ann", lambda p: p.metrics.bytes_ann),
                ("bytes_snn", lambda p: p.metrics.bytes_snn), ("energy_ratio", lambda p: p.metrics.energy_ratio),
            ):  # fmt: skip
                row[key] = mean(getter)
            rows.append(row)
        return rows

    @classmethod
    def write_summary_csv(cls, results: list[PointResult], path: str | Path) -> Path:
        rows = cls.summary_rows(results)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return path

    @classmethod
    def write_summary_markdown(cls, results: list[PointResult], path: str | Path) -> Path:
        """The table as the report shows it: `mean ± std`, the gap with its 95 % interval and p-value."""

        def pm(mean, std, digits=3):
            return f"{mean:.{digits}f}" + (f" ± {std:.{digits}f}" if std is not None else "")

        def gap(row):
            if row["auc_gap_p"] is None:
                return f"{row['auc_gap']:.3f} (1 seed)"
            return f"{row['auc_gap']:.3f} [{row['auc_gap_ci_low']:.3f}, {row['auc_gap_ci_high']:.3f}], p = {row['auc_gap_p']:.2g}"

        header = ["Configuration", "Seeds", "ROC-AUC ANN", "ROC-AUC SNN", "Gap ANN-SNN [95 % CI], Wilcoxon p",
                  "MACs", "SOPs", "r", "r*", "Cost ratio", "Energy ratio"]  # fmt: skip
        lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
        for r in cls.summary_rows(results):
            lines.append(
                f"| {r['config_id']} | {r['n_seeds']} | {pm(r['ann_auc_mean'], r['ann_auc_std'])} | {pm(r['snn_auc_mean'], r['snn_auc_std'])} "
                f"| {gap(r)} | {r['n_mac']:.0f} | {r['n_sop']:.0f} | {r['r']:.3f} | {r['r_critical']:.3f} "
                f"| {r['cost_ratio']:.3f} | {r['energy_ratio']:.2g} |"
            )
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path
