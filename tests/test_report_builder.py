import csv
import json

import pytest

from mac_sop_comparator.comparison import CSV_COLUMNS, EfficiencyMetrics, PointResult, ReportBuilder
from mac_sop_comparator.config import ArchitectureParams
from mac_sop_comparator.profiling import OpReport

PARAMS = ArchitectureParams(weight_bits=32, e_mac=1.0, e_ac=0.2, e_upd=0.3, memory_scenarios=["von_neumann"])
SIZES = [10, 5, 2]
PNG = b"\x89PNG\r\n\x1a\n"


def make_point(T=4, fingerprint="morgan", n_bits=1024, sops=60.0, ann_auc=0.74, snn_auc=0.70):
    ann = OpReport("ann", macs=100)
    snn = OpReport("snn", sops=sops, state_updates=20.0, spikes_per_layer=[8.0, 3.0, 1.0])
    metrics = EfficiencyMetrics.compute(ann, snn, PARAMS, T=T, layer_sizes=SIZES)
    return PointResult(
        point_id=f"{fingerprint}{n_bits}_T{T}", fingerprint=fingerprint, n_bits=n_bits, T=T,
        ann_auc=ann_auc, snn_auc=snn_auc, ann_balanced_accuracy=0.68, snn_balanced_accuracy=0.66,
        ann_report=ann, snn_report=snn, metrics=metrics, params=PARAMS,
    )  # fmt: skip


def sweep():
    return [make_point(T=4), make_point(T=8, sops=120.0), make_point(T=4, fingerprint="maccs", n_bits=166)]


def test_rows_are_wide_one_per_point():
    rows = ReportBuilder.rows(sweep())
    assert len(rows) == 3
    assert list(rows[0]) == CSV_COLUMNS
    assert rows[0]["point_id"] == "morgan1024_T4"
    assert rows[0]["cost_snn"] == pytest.approx(18.0)
    assert rows[0]["e_ac"] == 0.2 and rows[0]["snn_cheaper"] is True


def test_csv_round_trip(tmp_path):
    path = ReportBuilder.write_csv(sweep(), tmp_path / "out" / "metrics.csv")
    with path.open() as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0]) == CSV_COLUMNS
    assert [r["point_id"] for r in rows] == ["morgan1024_T4", "morgan1024_T8", "maccs166_T4"]
    assert float(rows[1]["r"]) == pytest.approx(120 / (8 * 100))
    assert rows[0]["snn_cheaper"] == "True"


def test_json_has_both_opreports_and_metrics(tmp_path):
    data = json.loads(ReportBuilder.write_json(sweep(), tmp_path / "opreports.json").read_text())
    assert [d["point_id"] for d in data] == ["morgan1024_T4", "morgan1024_T8", "maccs166_T4"]
    first = data[0]
    assert OpReport.from_dict(first["ann"]).macs == 100
    assert OpReport.from_dict(first["snn"]).sops == 60.0
    assert first["metrics"]["rate_per_layer"] == pytest.approx([8 / 40, 3 / 20, 1 / 8])


@pytest.mark.parametrize("results", [sweep(), sweep()[:1]], ids=["sweep", "single point"])
@pytest.mark.parametrize("plot", ["plot_operations", "plot_auc_vs_cost", "plot_rates"])
def test_plots_write_png(tmp_path, plot, results):
    path = getattr(ReportBuilder, plot)(results, tmp_path / "sub" / "fig.png")
    assert path.read_bytes().startswith(PNG)


def test_build_all(tmp_path):
    paths = ReportBuilder.build_all(sweep(), tmp_path)
    assert set(paths) == {"csv", "json", "operations", "auc_vs_cost", "rates"}
    assert all(p.exists() and p.parent == tmp_path for p in paths.values())
    assert {p.name for p in paths.values()} == {"metrics.csv", "opreports.json", "operations.png", "auc_vs_cost.png", "rates.png"}


def test_group_joins_points_that_differ_only_in_t():
    assert make_point(T=4).group == make_point(T=8).group == "morgan1024"
    assert make_point(fingerprint="maccs", n_bits=166).group == "maccs166"


@pytest.mark.parametrize(
    "call",
    [
        lambda b, p: b.rows([]),
        lambda b, p: b.write_csv([], p / "a.csv"),
        lambda b, p: b.write_json([], p / "a.json"),
        lambda b, p: b.plot_operations([], p / "a.png"),
        lambda b, p: b.plot_auc_vs_cost([], p / "a.png"),
        lambda b, p: b.plot_rates([], p / "a.png"),
        lambda b, p: b.build_all([], p),
    ],
)
def test_empty_results_are_rejected(tmp_path, call):
    with pytest.raises(ValueError, match="no results"):
        call(ReportBuilder, tmp_path)
