import json

import pytest
import yaml
from conftest import CONFIG

from mac_sop_comparator.experiments.cli import main

def test_prepare_writes_artifacts_and_log(workdir, capsys):
    assert main(["prepare", "--config", "c.yaml", "--results-dir", "out"]) == 0
    data_dir = workdir / "out" / "demo" / "data" / "morgan1024_r2_s42"
    assert (data_dir / "dataset.npz").is_file()
    assert json.loads((data_dir / "summary.json").read_text())["n_molecules"] == 200
    assert "prepared 200 molecules" in capsys.readouterr().out
    log = (workdir / "out" / "demo" / "run.log").read_text()
    assert "experiment 'demo': seed 42, 2 CPU threads" in log and "torch " in log


def test_results_dir_defaults_to_results(workdir):
    assert main(["prepare", "--config", "c.yaml"]) == 0
    assert (workdir / "results" / "demo" / "data" / "morgan1024_r2_s42" / "dataset.npz").is_file()


def test_missing_config_file(workdir, capsys):
    assert main(["prepare", "--config", "nope.yaml"]) == 2
    assert "not found" in capsys.readouterr().err


def test_invalid_config_names_the_field(workdir, capsys):
    bad = {**CONFIG, "model": {**CONFIG["model"], "snn": {"T": 0, "beta": 0.9, "threshold": 1.0}}}
    (workdir / "bad.yaml").write_text(yaml.safe_dump(bad))
    assert main(["prepare", "--config", "bad.yaml"]) == 2
    assert "model.snn.T" in capsys.readouterr().err


def test_dataset_not_available_is_a_clean_error(workdir, capsys):
    (workdir / "data_raw" / "tox21.csv.gz").unlink()  # nothing cached, and the URL does not resolve
    assert main(["prepare", "--config", "c.yaml"]) == 1
    assert "save it manually" in capsys.readouterr().err


def test_sweep_without_a_sweep_section_is_a_config_error(workdir, capsys):
    assert main(["sweep", "--config", "c.yaml"]) == 2
    assert "no 'sweep' section" in capsys.readouterr().err


# --- train / profile (ANN) ---------------------------------------------------------------------


@pytest.fixture
def prepared(workdir):
    """workdir plus a CPU config (`cpu.yaml`) and the data already prepared into ./out."""
    cpu = {**CONFIG, "experiment": {**CONFIG["experiment"], "device": "cpu"}, "training": {**CONFIG["training"], "epochs": 2}}
    (workdir / "cpu.yaml").write_text(yaml.safe_dump(cpu))
    assert main(["prepare", "--config", "cpu.yaml", "--results-dir", "out"]) == 0
    return workdir


def test_train_ann_writes_checkpoint_metrics_and_opreport(prepared, capsys):
    assert main(["train", "--model", "ann", "--config", "cpu.yaml", "--results-dir", "out"]) == 0
    point = prepared / "out" / "demo" / "morgan1024_r2_h8_T4_s42"
    assert (point / "ann.pt").is_file() and (point / "ann_curve.png").stat().st_size > 0
    metrics = json.loads((point / "ann_metrics.json").read_text())
    assert metrics["topology"] == [1024, 8, 2] and metrics["device"] == "cpu" and metrics["epochs_run"] == 2
    assert set(metrics["validation"]) == set(metrics["test"]) == {"roc_auc", "balanced_accuracy"}
    assert len(metrics["history"]) == 2
    report = json.loads((point / "opreport_ann.json").read_text())
    assert report["architecture"] == "ann" and report["per_inference"]["macs"] == 1024 * 8 + 8 * 2
    out = capsys.readouterr().out
    assert "trained the ANN" in out and "8208 MACs" in out


def test_train_is_reproducible(prepared):
    def run():
        assert main(["train", "--model", "ann", "--config", "cpu.yaml", "--results-dir", "out"]) == 0
        return json.loads((prepared / "out" / "demo" / "morgan1024_r2_h8_T4_s42" / "ann_metrics.json").read_text())["history"]

    assert run() == run()


def test_profile_after_train(prepared, capsys):
    main(["train", "--model", "ann", "--config", "cpu.yaml", "--results-dir", "out"])
    capsys.readouterr()
    assert main(["profile", "--config", "cpu.yaml", "--results-dir", "out"]) == 0
    assert "ann: 8208 MACs, 10 biases" in capsys.readouterr().out


def test_train_without_prepared_data_says_what_to_run(workdir, capsys):
    assert main(["train", "--model", "ann", "--config", "c.yaml", "--results-dir", "out"]) == 1
    assert "prepare" in capsys.readouterr().err


def test_profile_without_a_trained_model_says_what_to_run(prepared, capsys):
    assert main(["profile", "--config", "cpu.yaml", "--results-dir", "out"]) == 1
    assert "'train' command" in capsys.readouterr().err


def test_train_snn_writes_its_artifacts(prepared, capsys):
    assert main(["train", "--model", "snn", "--config", "cpu.yaml", "--results-dir", "out"]) == 0
    point = prepared / "out" / "demo" / "morgan1024_r2_h8_T4_s42"
    assert (point / "snn.pt").is_file() and (point / "snn_curve.png").stat().st_size > 0
    metrics = json.loads((point / "snn_metrics.json").read_text())
    assert metrics["model"] == "snn" and metrics["snn"] == {"T": 4, "beta": 0.9, "threshold": 1.0, "reset": "subtract"}
    report = json.loads((point / "opreport_snn.json").read_text())
    assert report["architecture"] == "snn"
    per = report["per_inference"]
    assert per["macs"] is None and per["state_updates"] == 4 * (8 + 2) and len(per["spikes_per_layer"]) == 3
    assert "trained the SNN" in capsys.readouterr().out


def test_ann_and_snn_start_from_the_same_initial_weights(prepared):
    from mac_sop_comparator.models import load_checkpoint

    for kind in ("ann", "snn"):
        assert main(["train", "--model", kind, "--config", "cpu.yaml", "--results-dir", "out"]) == 0
    point = prepared / "out" / "demo" / "morgan1024_r2_h8_T4_s42"
    assert [h["epoch"] for h in json.loads((point / "snn_metrics.json").read_text())["history"]] == [1, 2]
    ann, snn = load_checkpoint(point / "ann.pt"), load_checkpoint(point / "snn.pt")
    assert [p.shape for p in ann.parameters()] == [p.shape for p in snn.parameters()]


def test_profile_reports_both_models_when_both_are_trained(prepared, capsys):
    for kind in ("ann", "snn"):
        main(["train", "--model", kind, "--config", "cpu.yaml", "--results-dir", "out"])
    capsys.readouterr()
    assert main(["profile", "--config", "cpu.yaml", "--results-dir", "out"]) == 0
    out = capsys.readouterr().out
    assert "ann: 8208 MACs" in out and "snn: " in out and "SOPs" in out
    assert "40 state updates" in out  # T * (8 + 2) = 4 * 10


def test_separate_commands_find_each_others_artifacts(prepared, capsys):
    """prepare -> train ann -> train snn -> profile run as separate commands; the point is then rebuilt from files."""
    from mac_sop_comparator.config import load_config
    from mac_sop_comparator.experiments.layout import ResultsLayout
    from mac_sop_comparator.experiments.stages import load_point_result

    config = load_config("cpu.yaml")
    layout = ResultsLayout("out", config)
    assert not layout.point_dir.exists()
    for kind in ("ann", "snn"):
        assert main(["train", "--model", kind, "--config", "cpu.yaml", "--results-dir", "out"]) == 0
    assert main(["profile", "--config", "cpu.yaml", "--results-dir", "out"]) == 0
    for kind in ("ann", "snn"):
        assert all(path.is_file() for path in (layout.model_path(kind), layout.metrics_path(kind), layout.curve_path(kind), layout.opreport_path(kind)))
    assert layout.log_path.is_file() and (layout.data_dir / "dataset.npz").is_file()
    point = load_point_result(config, "out")
    assert point.point_id == layout.point_id == "morgan1024_r2_h8_T4_s42"
    assert point.metrics.n_mac == 1024 * 8 + 8 * 2 and point.T == 4


def test_compare_summarizes_and_writes_the_reports(prepared, capsys):
    for kind in ("ann", "snn"):
        main(["train", "--model", kind, "--config", "cpu.yaml", "--results-dir", "out"])
    capsys.readouterr()
    assert main(["compare", "--config", "cpu.yaml", "--results-dir", "out"]) == 0
    out = capsys.readouterr().out
    assert "point morgan1024_r2_h8_T4_s42 (T = 4)" in out and "ROC-AUC (test)" in out
    assert "8208 MACs" in out and "40 state updates" in out and "critical r* = " in out
    assert "costs use the energies of the config (e_mac 1, e_ac 0.2, e_upd 0.3)" in out
    report = prepared / "out" / "demo" / "morgan1024_r2_h8_T4_s42" / "report"
    assert report.name in out
    assert sorted(p.name for p in report.iterdir()) == [
        "auc_vs_cost.png", "metrics.csv", "operations.png", "opreports.json", "rates.png",
    ]  # fmt: skip


def test_compare_without_the_trained_models_is_a_clean_error(prepared, capsys):
    main(["train", "--model", "ann", "--config", "cpu.yaml", "--results-dir", "out"])
    capsys.readouterr()
    assert main(["compare", "--config", "cpu.yaml", "--results-dir", "out"]) == 1
    err = capsys.readouterr().err
    assert "lacks" in err and "snn.pt" in err and "train and profile both models first" in err


def test_run_does_everything_and_prints_the_summary(workdir, capsys):
    cpu = {**CONFIG, "experiment": {**CONFIG["experiment"], "device": "cpu"}, "training": {**CONFIG["training"], "epochs": 2}}
    (workdir / "cpu.yaml").write_text(yaml.safe_dump(cpu))
    assert main(["run", "--config", "cpu.yaml", "--results-dir", "out"]) == 0
    out = capsys.readouterr().out
    assert "point morgan1024_r2_h8_T4_s42 (T = 4)" in out and "SNN" in out and "reports -> " in out
    point = workdir / "out" / "demo" / "morgan1024_r2_h8_T4_s42"
    assert (point / "ann.pt").is_file() and (point / "snn.pt").is_file() and (point / "report" / "metrics.csv").is_file()
    log = (workdir / "out" / "demo" / "run.log").read_text()
    assert "state: idle -> configured" in log and "state: reporting -> idle" in log


def test_run_reports_the_failing_stage(workdir, capsys):
    (workdir / "data_raw" / "tox21.csv.gz").unlink()  # the data stage cannot get the dataset
    assert main(["run", "--config", "c.yaml", "--results-dir", "out"]) == 1
    err = capsys.readouterr().err
    assert "error: preparing data failed" in err and "save it manually" in err
    log = (workdir / "out" / "demo" / "run.log").read_text()
    assert "state: configured -> error" in log and "state: error -> idle" in log and "preparing data failed" in log


def test_run_with_an_invalid_config_exits_with_2(workdir, capsys):
    bad = {**CONFIG, "model": {**CONFIG["model"], "snn": {"T": 0, "beta": 0.9, "threshold": 1.0}}}
    (workdir / "bad.yaml").write_text(yaml.safe_dump(bad))
    assert main(["run", "--config", "bad.yaml"]) == 2


# sweep -----------------------------------------------------------------------------------------------


def sweep_config(workdir, sweep, name="sweep.yaml"):
    import yaml
    from conftest import CONFIG

    raw = {**CONFIG, "experiment": {**CONFIG["experiment"], "device": "cpu"}, "training": {**CONFIG["training"], "epochs": 2}, "sweep": sweep}
    (workdir / name).write_text(yaml.safe_dump(raw))
    return name


def test_sweep_runs_the_points_and_summarizes(workdir, capsys):
    name = sweep_config(workdir, {"T": [2, 4], "fingerprint": ["morgan"]})
    assert main(["sweep", "--config", name, "--results-dir", "out"]) == 0
    out = capsys.readouterr().out
    assert "sweep 'demo': 2 points (2 run, 0 skipped, 0 failed); ANN copied from another T in 1" in out
    assert "Pareto frontier:" in out and "models by compute cost" in out and "by total energy" in out
    assert "statistics" not in out  # one seed
    sweep_dir = workdir / "out" / "demo" / "sweep"
    assert (sweep_dir / "points.csv").is_file() and (sweep_dir / "pareto.csv").is_file()
    assert f"results -> {sweep_dir.relative_to(workdir)}" in out


def test_sweep_rerun_skips_and_force_runs_again(workdir, capsys):
    name = sweep_config(workdir, {"T": [2, 4], "fingerprint": ["morgan"]})
    main(["sweep", "--config", name, "--results-dir", "out"])
    capsys.readouterr()
    assert main(["sweep", "--config", name, "--results-dir", "out"]) == 0
    assert "(0 run, 2 skipped, 0 failed)" in capsys.readouterr().out
    assert main(["sweep", "--config", name, "--results-dir", "out", "--force"]) == 0
    assert "(2 run, 0 skipped, 0 failed)" in capsys.readouterr().out


def test_sweep_with_seeds_reports_the_statistics(workdir, capsys):
    name = sweep_config(workdir, {"T": [2], "fingerprint": ["morgan"], "seeds": [1, 2]})
    assert main(["sweep", "--config", name, "--results-dir", "out"]) == 0
    assert "seed statistics -> statistics.csv, statistics.png" in capsys.readouterr().out
    assert (workdir / "out" / "demo" / "sweep" / "statistics.csv").is_file()


def test_sweep_with_too_many_points_is_a_config_error(workdir, capsys):
    name = sweep_config(workdir, {"T": [2, 4, 8], "fingerprint": ["morgan"], "max_points": 2})
    assert main(["sweep", "--config", name, "--results-dir", "out"]) == 2
    assert "3 points, more than max_points = 2" in capsys.readouterr().err
    assert not (workdir / "out" / "demo" / "morgan1024_r2_h8_T2_s42").exists()  # nothing was trained


def test_sweep_with_a_failed_point_exits_with_1(workdir, capsys, monkeypatch):
    from mac_sop_comparator.experiments import sweep as sweep_module
    from mac_sop_comparator.experiments.runner import ExperimentRunner

    class Flaky(ExperimentRunner):
        def _train(self):
            if self.config.model.snn.T == 2:
                raise RuntimeError("boom")
            super()._train()

    monkeypatch.setattr(sweep_module, "ExperimentRunner", Flaky)
    name = sweep_config(workdir, {"T": [2, 4], "fingerprint": ["morgan"]})
    assert main(["sweep", "--config", name, "--results-dir", "out"]) == 1
    out = capsys.readouterr().out
    assert "(1 run, 0 skipped, 1 failed)" in out and "failed morgan1024_r2_h8_T2_s42: training failed: boom" in out


# analyze ---------------------------------------------------------------------------------------------


def test_analyze_without_a_sweep_section_is_a_config_error(workdir, capsys):
    assert main(["analyze", "--config", "c.yaml"]) == 2
    assert "no 'sweep' section" in capsys.readouterr().err


def test_analyze_before_any_sweep_asks_to_run_it(workdir, capsys):
    name = sweep_config(workdir, {"T": [2], "fingerprint": ["morgan"]})
    assert main(["analyze", "--config", name, "--results-dir", "out"]) == 1
    assert "run the 'sweep' command first" in capsys.readouterr().err


def test_analyze_writes_tables_and_figures_and_reports_the_test(workdir, capsys):
    name = sweep_config(workdir, {"T": [2, 4], "fingerprint": ["morgan"], "seeds": [1, 2, 3]})
    assert main(["sweep", "--config", name, "--results-dir", "out"]) == 0
    capsys.readouterr()
    assert main(["analyze", "--config", name, "--results-dir", "out"]) == 0
    out = capsys.readouterr().out
    assert "analysis of 'demo': 6 of 6 points" in out
    assert "morgan1024_h8_T2: ROC-AUC gap ANN-SNN" in out and "Wilcoxon p = " in out and "(n = 3)" in out
    analysis = workdir / "out" / "demo" / "analysis"
    assert {f.name for f in analysis.iterdir()} == {
        "summary.csv", "summary.md", "energy_breakdown.png", "cost_vs_T.png", "rates_vs_T.png", "comparison.csv", "accuracy_gap.png",
    }  # fmt: skip


def test_analyze_skips_the_test_when_every_configuration_has_one_seed(workdir, capsys):
    name = sweep_config(workdir, {"T": [2, 4], "fingerprint": ["morgan"]})
    main(["sweep", "--config", name, "--results-dir", "out"])
    capsys.readouterr()
    assert main(["analyze", "--config", name, "--results-dir", "out"]) == 0
    assert "no configuration has repeated seeds" in capsys.readouterr().out


def test_analyze_reports_the_points_that_have_no_results(workdir, capsys):
    name = sweep_config(workdir, {"T": [2], "fingerprint": ["morgan"], "seeds": [1]})
    main(["sweep", "--config", name, "--results-dir", "out"])
    bigger = sweep_config(workdir, {"T": [2, 4], "fingerprint": ["morgan"], "seeds": [1]}, name="bigger.yaml")
    capsys.readouterr()
    assert main(["analyze", "--config", bigger, "--results-dir", "out"]) == 0
    assert "1 of 2 points (1 missing or stale)" in capsys.readouterr().out
