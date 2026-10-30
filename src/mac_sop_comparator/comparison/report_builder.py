"""ReportBuilder: CSV of metrics, JSON of OpReports and the PNG figures of the comparison (D-38)."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt  # noqa: E402

from mac_sop_comparator.comparison.point import PointResult  # noqa: E402

CSV_COLUMNS = [
    "point_id", "fingerprint", "n_bits", "T",
    "ann_auc", "snn_auc", "ann_balanced_accuracy", "snn_balanced_accuracy",
    "n_mac", "n_sop", "n_upd", "r", "r_critical",
    "cost_ann", "cost_snn", "cost_ratio", "ops_ratio", "snn_cheaper",
    "e_mac", "e_ac", "e_upd",
]  # fmt: skip
COST_LABEL = "relative cost (illustrative energies)"
ANN_COLOR, SNN_COLOR = "tab:blue", "tab:orange"


def _require(results: list[PointResult]) -> None:
    if not results:
        raise ValueError("there are no results to report")


def _save(fig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


class ReportBuilder:
    """Turns a list of PointResult into the artifacts of the design: CSV, JSON and PNG."""

    @staticmethod
    def rows(results: list[PointResult]) -> list[dict]:
        """One wide row per point (D-38)."""
        _require(results)
        rows = []
        for p in results:
            m = p.metrics
            rows.append(
                dict(
                    point_id=p.point_id, fingerprint=p.fingerprint, n_bits=p.n_bits, T=p.T,
                    ann_auc=p.ann_auc, snn_auc=p.snn_auc,
                    ann_balanced_accuracy=p.ann_balanced_accuracy, snn_balanced_accuracy=p.snn_balanced_accuracy,
                    n_mac=m.n_mac, n_sop=m.n_sop, n_upd=m.n_upd, r=m.r, r_critical=m.r_critical,
                    cost_ann=m.cost_ann, cost_snn=m.cost_snn, cost_ratio=m.cost_ratio, ops_ratio=m.ops_ratio,
                    snn_cheaper=m.snn_cheaper,
                    e_mac=p.params.e_mac, e_ac=p.params.e_ac, e_upd=p.params.e_upd,
                )
            )  # fmt: skip
        return rows

    @classmethod
    def write_csv(cls, results: list[PointResult], path: str | Path) -> Path:
        rows = cls.rows(results)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        return path

    @staticmethod
    def write_json(results: list[PointResult], path: str | Path) -> Path:
        """OpReports of both models plus the unified metrics (with the per-layer rates) of every point."""
        _require(results)
        data = [
            {
                "point_id": p.point_id,
                "ann": p.ann_report.to_dict(),
                "snn": p.snn_report.to_dict(),
                "metrics": p.metrics.to_dict(),
            }
            for p in results
        ]
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2) + "\n")
        return path

    @staticmethod
    def plot_operations(results: list[PointResult], path: str | Path) -> Path:
        """MACs of the ANN against the SOPs of the SNN, with its state updates stacked on top."""
        _require(results)
        xs = range(len(results))
        width = 0.38
        fig, ax = plt.subplots(figsize=(max(5.0, 1.6 * len(results) + 2), 4))
        ax.bar([x - width / 2 for x in xs], [p.metrics.n_mac for p in results], width, color=ANN_COLOR, label="ANN MACs")
        ax.bar([x + width / 2 for x in xs], [p.metrics.n_sop for p in results], width, color=SNN_COLOR, label="SNN SOPs")
        ax.bar(
            [x + width / 2 for x in xs], [p.metrics.n_upd for p in results], width,
            bottom=[p.metrics.n_sop for p in results], color="tab:red", alpha=0.7, label="SNN state updates",
        )  # fmt: skip
        ax.set(xticks=list(xs), xticklabels=[p.point_id for p in results], ylabel="operations per inference")
        ax.tick_params(axis="x", rotation=30 if len(results) > 1 else 0)
        ax.set_title("Operations per inference")
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(top=ax.get_ylim()[1] * 1.15)  # room for the legend above the bars
        ax.legend(loc="upper center", ncol=3)
        return _save(fig, path)

    @staticmethod
    def plot_auc_vs_cost(results: list[PointResult], path: str | Path) -> Path:
        """Test ROC-AUC against relative cost, one marker per model and point (the Pareto input of phase 7)."""
        _require(results)
        fig, ax = plt.subplots(figsize=(6, 4.2))
        ax.scatter([p.metrics.cost_ann for p in results], [p.ann_auc for p in results], color=ANN_COLOR, marker="o", label="ANN")
        ax.scatter([p.metrics.cost_snn for p in results], [p.snn_auc for p in results], color=SNN_COLOR, marker="^", label="SNN")
        if len(results) <= 12:
            for p in results:
                ax.annotate(p.point_id, (p.metrics.cost_snn, p.snn_auc), fontsize=7, xytext=(4, 4), textcoords="offset points")
        ax.set(xscale="log", xlabel=COST_LABEL, ylabel="test ROC-AUC", title="Accuracy against cost")
        ax.grid(alpha=0.3)
        ax.legend()
        return _save(fig, path)

    @staticmethod
    def plot_rates(results: list[PointResult], path: str | Path) -> Path:
        """Effective rate r against the critical rate r*: the SNN is cheaper wherever r is below r*."""
        _require(results)
        fig, ax = plt.subplots(figsize=(6, 4.2))
        if len(results) == 1:
            m = results[0].metrics
            ax.bar(["r (measured)", "r* (critical)"], [m.r, m.r_critical], color=[SNN_COLOR, "gray"])
            ax.set_title(f"Effective against critical rate · {results[0].point_id}")
        else:
            groups = sorted({p.group for p in results})
            for i, group in enumerate(groups):
                pts = sorted((p for p in results if p.group == group), key=lambda p: p.T)
                color = f"C{i}"
                ts = [p.T for p in pts]
                ax.plot(ts, [p.metrics.r for p in pts], color=color, marker="o", label=f"{group} r")
                ax.plot(ts, [p.metrics.r_critical for p in pts], color=color, linestyle="--", label=f"{group} r*")
                ax.fill_between(ts, 0, [p.metrics.r_critical for p in pts], color=color, alpha=0.08)
            ax.set_xlabel("time steps T")
            ax.set_title("Effective against critical rate (shaded: SNN cheaper)")
            ax.legend(fontsize=7)
        ax.set_ylabel("rate")
        ax.grid(alpha=0.3)
        return _save(fig, path)

    @classmethod
    def build_all(cls, results: list[PointResult], directory: str | Path) -> dict[str, Path]:
        """Writes `metrics.csv`, `opreports.json` and the three figures into `directory`."""
        directory = Path(directory)
        return {
            "csv": cls.write_csv(results, directory / "metrics.csv"),
            "json": cls.write_json(results, directory / "opreports.json"),
            "operations": cls.plot_operations(results, directory / "operations.png"),
            "auc_vs_cost": cls.plot_auc_vs_cost(results, directory / "auc_vs_cost.png"),
            "rates": cls.plot_rates(results, directory / "rates.png"),
        }
