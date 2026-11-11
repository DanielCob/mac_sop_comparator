"""Figures and tables of the analysis (A17, D-50): efficiency against T, rates, energy breakdown, accuracy gap."""

import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt  # noqa: E402

from mac_sop_comparator.comparison.analysis import Analysis  # noqa: E402
from mac_sop_comparator.comparison.point import PointResult  # noqa: E402
from mac_sop_comparator.comparison.report_builder import ANN_COLOR, SNN_COLOR, _require, _save  # noqa: E402
from mac_sop_comparator.comparison.statistics import SeedStatistics  # noqa: E402

LINESTYLES = ["-", "--", ":", "-."]
MARKERS = ["o", "s", "^", "D", "v"]
MAIN_T = 10  # the value of T the energy breakdown shows when the sweep has it


def _fingerprint(p: PointResult) -> str:
    return f"{p.fingerprint}{p.n_bits}"


def _topology(p: PointResult) -> str:
    return "[" + ", ".join(map(str, p.hidden_layers)) + "]" if p.hidden_layers else "?"


def _by_t(results: list[PointResult]) -> dict[str, dict[int, list[PointResult]]]:
    """family -> T -> points (several when the seeds are repeated)."""
    grouped: dict[str, dict[int, list[PointResult]]] = {}
    for p in results:
        grouped.setdefault(p.family, {}).setdefault(p.T, []).append(p)
    return grouped


def _style(results: list[PointResult]):
    fingerprints = sorted({_fingerprint(p) for p in results})
    topologies = sorted({_topology(p) for p in results}, key=lambda t: (len(t), t))
    return (
        {f: f"C{i}" for i, f in enumerate(fingerprints)},
        {t: (LINESTYLES[i % len(LINESTYLES)], MARKERS[i % len(MARKERS)]) for i, t in enumerate(topologies)},
    )


def _mean(points: list[PointResult], getter) -> float:
    return statistics.fmean(getter(p) for p in points)


class AnalysisFigures:
    @staticmethod
    def plot_cost_vs_t(results: list[PointResult], path: str | Path) -> Path:
        """Compute-only cost ratio C_SNN / C_ANN against T: below 1 the SNN is cheaper."""
        _require(results)
        colors, styles = _style(results)
        fig, ax = plt.subplots(figsize=(8.4, 4.4))
        for family, by_t in sorted(_by_t(results).items()):
            sample = next(iter(by_t.values()))[0]
            ts = sorted(by_t)
            linestyle, marker = styles[_topology(sample)]
            ax.plot(ts, [_mean(by_t[t], lambda p: p.metrics.cost_ratio) for t in ts], color=colors[_fingerprint(sample)],
                    linestyle=linestyle, marker=marker, label=family)  # fmt: skip
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
        ax.text(max(p.T for p in results), 1.06, "ANN cheaper above this line", ha="right", va="bottom", fontsize=7, color="gray")
        ax.set(yscale="log", xlabel="time steps T", ylabel="cost ratio C_SNN / C_ANN", title="Relative cost of the SNN against T")
        ax.set_ylim(top=1.6)
        ax.set_xticks(sorted({p.T for p in results}))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0))  # outside: the lines fill the plot
        return _save(fig, path)

    @staticmethod
    def plot_rates_vs_t(results: list[PointResult], path: str | Path) -> Path:
        """Effective rate r (solid) and critical rate r* (dashed) against T, one panel per fingerprint."""
        _require(results)
        _, styles = _style(results)
        topologies = list(styles)  # one color per topology: the panels already separate the fingerprints
        fingerprints = sorted({_fingerprint(p) for p in results})
        fig, axes = plt.subplots(1, len(fingerprints), figsize=(4.2 * len(fingerprints), 4), sharey=True, squeeze=False)
        for ax, fp in zip(axes[0], fingerprints):
            for family, by_t in sorted(_by_t([p for p in results if _fingerprint(p) == fp]).items()):
                sample = next(iter(by_t.values()))[0]
                ts = sorted(by_t)
                _, marker = styles[_topology(sample)]
                color = f"C{topologies.index(_topology(sample))}"
                ax.plot(ts, [_mean(by_t[t], lambda p: p.metrics.r) for t in ts], color=color, marker=marker, label=f"{_topology(sample)} r")
                ax.plot(ts, [_mean(by_t[t], lambda p: p.metrics.r_critical) for t in ts], color="gray", linestyle="--", linewidth=1,
                        label="r* (critical)" if "r* (critical)" not in ax.get_legend_handles_labels()[1] else None)  # fmt: skip
            ax.set(yscale="log", xlabel="time steps T", title=fp)
            ax.set_xticks(sorted({p.T for p in results}))
            ax.grid(alpha=0.3)
        axes[0][0].set_ylabel("rate")
        axes[0][0].legend(fontsize=6)
        fig.suptitle("Effective against critical rate: the SNN is cheaper where r < r*")
        return _save(fig, path)

    @staticmethod
    def plot_energy_breakdown(results: list[PointResult], path: str | Path, T: int | None = None) -> Path:
        """Compute and memory energy of both networks for each configuration at one T (D-37)."""
        _require(results)
        ts = sorted({p.T for p in results})
        T = T if T is not None else (MAIN_T if MAIN_T in ts else ts[len(ts) // 2])
        chosen = [p for p in results if p.T == T]
        if not chosen:
            raise ValueError(f"no result has T = {T}; available: {ts}")
        families = {}
        for p in chosen:
            families.setdefault(p.family, []).append(p)
        fig, ax = plt.subplots(figsize=(max(6.0, 0.95 * len(families) + 2), 4.4))
        width = 0.2
        series = [
            ("ANN compute", ANN_COLOR, "", lambda p: p.metrics.cost_ann),
            ("ANN memory", ANN_COLOR, "//", lambda p: p.metrics.energy_ann - p.metrics.cost_ann),
            ("SNN compute", SNN_COLOR, "", lambda p: p.metrics.cost_snn),
            ("SNN memory", SNN_COLOR, "//", lambda p: p.metrics.energy_snn - p.metrics.cost_snn),
        ]
        for i, (name, color, hatch, getter) in enumerate(series):
            ax.bar([x + (i - 1.5) * width for x in range(len(families))], [_mean(pts, getter) for pts in families.values()], width,
                   color=color, hatch=hatch, edgecolor="white" if not hatch else "black", linewidth=0.4, label=name)  # fmt: skip
        ax.set(yscale="log", ylabel="relative energy (one MAC = 1)", title=f"Energy per inference, compute and memory (T = {T})")
        ax.set_xticks(range(len(families)))
        ax.set_xticklabels(list(families), rotation=30, ha="right", fontsize=7)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=7, ncol=2)
        return _save(fig, path)

    @staticmethod
    def plot_accuracy_gap(results: list[PointResult], path: str | Path) -> Path:
        """ROC-AUC of both networks per repeated configuration (mean ± std) with the gap, its interval and the test."""
        stats_rows = {r["config_id"]: r for r in SeedStatistics.aggregate(results) if r["n_seeds"] > 1}
        comparisons = {c.config_id: c for c in Analysis.comparisons(results, "roc_auc")}
        if not stats_rows:
            raise ValueError("no configuration has more than one seed")
        ids = list(stats_rows)
        fig, ax = plt.subplots(figsize=(max(5.0, 2.4 * len(ids) + 2), 4.6))
        for offset, key, color, name in ((-0.12, "ann_auc", ANN_COLOR, "ANN"), (0.12, "snn_auc", SNN_COLOR, "SNN")):
            ax.errorbar([i + offset for i in range(len(ids))], [stats_rows[c][f"{key}_mean"] for c in ids],
                        [stats_rows[c][f"{key}_std"] for c in ids], fmt="o", color=color, capsize=4, label=name)  # fmt: skip
        top = max(stats_rows[c]["ann_auc_mean"] + stats_rows[c]["ann_auc_std"] for c in ids)
        for i, c in enumerate(ids):
            cmp = comparisons[c]
            ax.annotate(f"gap {cmp.diff_mean:.3f}\n[{cmp.ci_low:.3f}, {cmp.ci_high:.3f}]\np = {cmp.p_value:.2g}\n(n = {cmp.n_seeds})",
                        (i, top), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=7)  # fmt: skip
        ax.set_ylim(top=top + 0.05 * (top - min(stats_rows[c]["snn_auc_mean"] for c in ids)) + 0.03)
        ax.set(xticks=range(len(ids)), xticklabels=ids, ylabel="test ROC-AUC", title="Accuracy of both networks over seeds (mean ± std)")
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(loc="lower left")
        return _save(fig, path)


def build_analysis(results: list[PointResult], directory: str | Path) -> dict[str, Path]:
    """Writes every table and figure the results allow into `directory`; what needs several T or seeds is skipped."""
    _require(results)
    directory = Path(directory)
    paths = {
        "summary_csv": Analysis.write_summary_csv(results, directory / "summary.csv"),
        "summary_md": Analysis.write_summary_markdown(results, directory / "summary.md"),
        "energy_breakdown": AnalysisFigures.plot_energy_breakdown(results, directory / "energy_breakdown.png"),
    }
    if len({p.T for p in results}) > 1:
        paths["cost_vs_t"] = AnalysisFigures.plot_cost_vs_t(results, directory / "cost_vs_T.png")
        paths["rates_vs_t"] = AnalysisFigures.plot_rates_vs_t(results, directory / "rates_vs_T.png")
    if Analysis.comparisons(results):
        paths["comparison_csv"] = Analysis.write_comparisons(results, directory / "comparison.csv")
        paths["accuracy_gap"] = AnalysisFigures.plot_accuracy_gap(results, directory / "accuracy_gap.png")
    return paths
