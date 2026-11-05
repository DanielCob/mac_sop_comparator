import csv
import math
import statistics as st

import pytest
from test_pareto import point

from mac_sop_comparator.comparison import ReportBuilder, SeedStatistics
from mac_sop_comparator.comparison.statistics import COLUMNS


def test_mean_and_sample_std_over_the_seeds_by_hand():
    rows = SeedStatistics.aggregate([point(seed=1, snn_auc=0.60), point(seed=2, snn_auc=0.70), point(seed=3, snn_auc=0.80)])
    (row,) = rows
    assert (row["config_id"], row["n_seeds"], row["seeds"]) == ("morgan10_h5_T4", 3, "1;2;3")
    assert row["snn_auc_mean"] == pytest.approx(0.70)
    assert row["snn_auc_std"] == pytest.approx(0.10)  # sample std of 0.6, 0.7, 0.8 (n - 1)
    assert row["ann_auc_std"] == pytest.approx(0.0)


def test_the_auc_gap_is_computed_per_seed_before_aggregating():
    rows = SeedStatistics.aggregate([point(seed=1, ann_auc=0.80, snn_auc=0.70), point(seed=2, ann_auc=0.60, snn_auc=0.58)])
    gaps = [0.10, 0.02]
    assert rows[0]["auc_gap_mean"] == pytest.approx(st.fmean(gaps))
    assert rows[0]["auc_gap_std"] == pytest.approx(st.stdev(gaps))


def test_one_seed_has_a_blank_std():
    (row,) = SeedStatistics.aggregate([point()])
    assert row["n_seeds"] == 1 and row["snn_auc_std"] == "" and row["snn_auc_mean"] == 0.70
    assert not SeedStatistics.has_repeats([row])


def test_configurations_differing_in_t_or_topology_stay_apart():
    rows = SeedStatistics.aggregate([point(T=4, seed=1), point(T=8, seed=1), point(hidden=(16,), seed=1), point(T=4, seed=2)])
    assert [(r["config_id"], r["n_seeds"]) for r in rows] == [("morgan10_h5_T4", 2), ("morgan10_h5_T8", 1), ("morgan10_h16_T4", 1)]


def test_derived_metrics_are_aggregated_too():
    # a second seed with more SOPs: r = SOPs / (T * N_MAC) = 60/400 and 120/400
    rows = SeedStatistics.aggregate([point(seed=1, sops=60.0), point(seed=2, sops=120.0)])
    assert rows[0]["r_mean"] == pytest.approx((0.15 + 0.30) / 2)
    assert rows[0]["n_sop_mean"] == pytest.approx(90.0)
    assert math.isfinite(rows[0]["energy_ratio_mean"])


def test_csv_has_the_documented_columns(tmp_path):
    path = SeedStatistics.write_csv([point(seed=1), point(seed=2)], tmp_path / "statistics.csv")
    with path.open() as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == COLUMNS
        assert len(list(reader)) == 1


def test_empty_results_are_an_error(tmp_path):
    with pytest.raises(ValueError, match="no results"):
        SeedStatistics.aggregate([])


def test_the_plot_needs_a_repeated_configuration(tmp_path):
    with pytest.raises(ValueError, match="more than one seed"):
        ReportBuilder.plot_statistics([point(seed=1), point(seed=1, T=8)], tmp_path / "s.png")
    path = ReportBuilder.plot_statistics([point(seed=1), point(seed=2)], tmp_path / "s.png")
    assert path.is_file() and path.stat().st_size > 0


def test_build_sweep_adds_statistics_only_when_a_seed_repeats(tmp_path):
    assert "statistics" not in ReportBuilder.build_sweep([point(seed=1), point(seed=1, T=8)], tmp_path / "a")
    paths = ReportBuilder.build_sweep([point(seed=1), point(seed=2)], tmp_path / "b")
    assert paths["statistics"].name == "statistics.csv" and paths["statistics_plot"].name == "statistics.png"


def test_point_config_id_drops_the_seed():
    assert point(seed=9).config_id == "morgan10_h5_T4" and point(seed=9).group.endswith("_s9")
