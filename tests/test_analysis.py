import csv

import pytest
from test_pareto import point

from mac_sop_comparator.comparison import Analysis, AnalysisFigures, build_analysis, paired_comparison


def seeds(ann_aucs, snn_aucs, **kw):
    return [point(seed=i, ann_auc=a, snn_auc=s, **kw) for i, (a, s) in enumerate(zip(ann_aucs, snn_aucs), start=1)]


# Paired comparison -------------------------------------------------------------------------------


def test_paired_comparison_by_hand():
    # differences 0.02, 0.03, 0.01, 0.04: mean 0.025, sd = sqrt(0.0005 / 3) = 0.012910, t(0.975, 3) = 3.1824
    c = paired_comparison("cfg", [0.80, 0.82, 0.78, 0.84], [0.78, 0.79, 0.77, 0.80], "roc_auc")
    assert (c.n_seeds, c.ann_mean, c.snn_mean) == (4, pytest.approx(0.81), pytest.approx(0.785))
    assert c.diff_mean == pytest.approx(0.025)
    half = 3.1824463 * 0.0129099 / 2
    assert (c.ci_low, c.ci_high) == pytest.approx((0.025 - half, 0.025 + half), abs=1e-6)
    assert c.p_value == pytest.approx(0.125)  # all four differences positive: 2 / 2^4


def test_five_seeds_can_never_reach_significance_but_six_can():
    # the reason D-50 asks for 30 seeds: the smallest two-sided Wilcoxon p-value is 2 / 2^n
    five = paired_comparison("c", [0.8] * 5, [0.7, 0.71, 0.72, 0.73, 0.74], "roc_auc")
    six = paired_comparison("c", [0.8] * 6, [0.7, 0.71, 0.72, 0.73, 0.74, 0.75], "roc_auc")
    assert five.p_value == pytest.approx(0.0625) and five.p_value > 0.05
    assert six.p_value == pytest.approx(0.03125) and six.p_value < 0.05


def test_identical_models_have_p_one_and_a_zero_gap():
    c = paired_comparison("c", [0.7, 0.8, 0.9], [0.7, 0.8, 0.9], "roc_auc")
    assert (c.diff_mean, c.p_value, c.ci_low, c.ci_high) == (0.0, 1.0, 0.0, 0.0)


def test_a_snn_better_than_the_ann_gives_a_negative_gap():
    assert paired_comparison("c", [0.7, 0.71, 0.72], [0.8, 0.81, 0.82], "roc_auc").diff_mean == pytest.approx(-0.1)


@pytest.mark.parametrize("ann,snn", [([0.7], [0.6]), ([0.7, 0.8], [0.6]), ([], [])])
def test_a_pair_needs_at_least_two_matching_seeds(ann, snn):
    with pytest.raises(ValueError, match="at least 2"):
        paired_comparison("c", ann, snn, "roc_auc")


def test_comparisons_group_by_configuration_and_skip_single_seeds():
    results = seeds([0.8, 0.82], [0.7, 0.72]) + [point(T=8, seed=1)] + seeds([0.9, 0.9], [0.8, 0.8], hidden=(16,))
    got = Analysis.comparisons(results)
    assert [(c.config_id, c.n_seeds) for c in got] == [("morgan10_h5_T4", 2), ("morgan10_h16_T4", 2)]
    assert got[0].diff_mean == pytest.approx(0.1)


def test_balanced_accuracy_is_compared_too():
    results = [point(seed=1), point(seed=2)]
    (c,) = Analysis.comparisons(results, "balanced_accuracy")
    assert c.metric == "balanced_accuracy" and c.diff_mean == 0.0  # the test fixture gives both 0.6
    with pytest.raises(ValueError, match="metric"):
        Analysis.comparisons(results, "f1")


def test_the_comparison_csv_lists_both_metrics(tmp_path):
    path = Analysis.write_comparisons(seeds([0.8, 0.82], [0.7, 0.72]), tmp_path / "c.csv")
    rows = list(csv.DictReader(path.open()))
    assert [r["metric"] for r in rows] == ["roc_auc", "balanced_accuracy"]
    with pytest.raises(ValueError, match="more than one seed"):
        Analysis.write_comparisons([point()], tmp_path / "x.csv")


# Summary table -----------------------------------------------------------------------------------


def test_summary_row_by_hand():
    (row,) = Analysis.summary_rows(seeds([0.80, 0.82, 0.78, 0.84], [0.78, 0.79, 0.77, 0.80]))
    assert (row["config_id"], row["n_seeds"]) == ("morgan10_h5_T4", 4)
    assert row["ann_auc_mean"] == pytest.approx(0.81) and row["snn_auc_mean"] == pytest.approx(0.785)
    assert row["auc_gap"] == pytest.approx(0.025) and row["auc_gap_p"] == pytest.approx(0.125)
    assert (row["n_mac"], row["n_sop"], row["n_upd"]) == (100, 60.0, 20.0)
    assert row["cost_ratio"] == pytest.approx(0.18) and row["bytes_ann"] == 468 and row["bytes_snn"] == 52


def test_a_single_seed_has_no_std_and_no_test():
    (row,) = Analysis.summary_rows([point()])
    assert row["ann_auc_std"] is None and row["auc_gap_p"] is None and row["auc_gap"] == pytest.approx(0.04)


def test_markdown_table_shows_mean_std_and_the_test(tmp_path):
    text = Analysis.write_summary_markdown(seeds([0.80, 0.82, 0.78, 0.84], [0.78, 0.79, 0.77, 0.80]) + [point(T=8)], tmp_path / "s.md").read_text()
    lines = text.strip().split("\n")
    assert lines[0].startswith("| Configuration | Seeds |") and lines[1].startswith("|---|")
    assert "| morgan10_h5_T4 | 4 | 0.810 ± 0.026 | 0.785 ± 0.013 | 0.025 [0.004, 0.046], p = 0.12 |" in lines[2]
    assert "(1 seed)" in lines[3]
    assert len(lines) == 4


def test_the_summary_csv_has_one_row_per_configuration(tmp_path):
    path = Analysis.write_summary_csv([point(T=4), point(T=8), point(T=4, seed=7)], tmp_path / "s.csv")
    assert [r["config_id"] for r in csv.DictReader(path.open())] == ["morgan10_h5_T4", "morgan10_h5_T8"]


def test_empty_results_are_an_error(tmp_path):
    with pytest.raises(ValueError, match="no results"):
        Analysis.summary_rows([])


# Figures -----------------------------------------------------------------------------------------


def sweep_points():
    return [point(T=T, hidden=h, ann_auc=0.74, snn_auc=0.70 + 0.002 * T, sops=30.0 * T) for h in ((16,), (128, 64)) for T in (4, 8, 16)]


def test_the_figures_of_a_sweep(tmp_path):
    for name, plot in (("cost.png", AnalysisFigures.plot_cost_vs_t), ("rates.png", AnalysisFigures.plot_rates_vs_t),
                       ("energy.png", AnalysisFigures.plot_energy_breakdown)):  # fmt: skip
        path = plot(sweep_points(), tmp_path / name)
        assert path.is_file() and path.stat().st_size > 0


def test_the_energy_breakdown_needs_an_existing_t(tmp_path):
    assert AnalysisFigures.plot_energy_breakdown(sweep_points(), tmp_path / "e.png", T=8).is_file()
    with pytest.raises(ValueError, match="no result has T = 99"):
        AnalysisFigures.plot_energy_breakdown(sweep_points(), tmp_path / "e.png", T=99)


def test_the_accuracy_gap_figure_needs_repeated_seeds(tmp_path):
    with pytest.raises(ValueError, match="more than one seed"):
        AnalysisFigures.plot_accuracy_gap([point()], tmp_path / "g.png")
    path = AnalysisFigures.plot_accuracy_gap(seeds([0.80, 0.82, 0.78], [0.78, 0.79, 0.77]), tmp_path / "g.png")
    assert path.is_file() and path.stat().st_size > 0


def test_build_analysis_writes_only_what_the_results_allow(tmp_path):
    single_seed = build_analysis(sweep_points(), tmp_path / "a")
    assert set(single_seed) == {"summary_csv", "summary_md", "energy_breakdown", "cost_vs_t", "rates_vs_t"}
    one_t = build_analysis(seeds([0.80, 0.82, 0.78], [0.78, 0.79, 0.77]), tmp_path / "b")  # one T, three seeds
    assert set(one_t) == {"summary_csv", "summary_md", "energy_breakdown", "comparison_csv", "accuracy_gap"}
    both = build_analysis(sweep_points() + seeds([0.8, 0.82], [0.7, 0.72]), tmp_path / "c")
    assert {"cost_vs_t", "rates_vs_t", "accuracy_gap", "comparison_csv"} <= set(both)
    assert all(p.is_file() for p in both.values())
