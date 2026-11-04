import json
from pathlib import Path

import pytest

from mac_sop_comparator.comparison import EfficiencyMetrics, PointResult
from mac_sop_comparator.config import parse_config
from mac_sop_comparator.experiments import runner as runner_module
from mac_sop_comparator.experiments.runner import TRANSITIONS, ExperimentError, ExperimentRunner, RunnerState as S
from mac_sop_comparator.experiments.layout import ResultsLayout
from mac_sop_comparator.experiments.stages import load_point_result
from mac_sop_comparator.profiling import OpReport

CONFIG = {
    "experiment": {"name": "demo", "seed": 42},
    "data": {
        "dataset": "tox21",
        "task": "TASK",
        "fingerprint": {"type": "morgan", "n_bits": 1024, "radius": 2},
        "split": {"train": 0.8, "val": 0.1, "test": 0.1},
    },
    "model": {"hidden_layers": [8], "n_outputs": 2, "snn": {"T": 4, "beta": 0.9, "threshold": 1.0}},
    "training": {"optimizer": "adam", "lr": 1e-3, "batch_size": 16, "epochs": 1},
    "architecture": {"weight_bits": 32, "e_mac": 1, "e_ac": 0.2, "e_upd": 0.3, "e_byte_von_neumann": 53, "e_byte_neuromorphic": 1.3},
}
STAGES = ["_prepare_data", "_train", "_evaluate", "_measure", "_compare", "_report"]
HAPPY_PATH = [S.IDLE, S.CONFIGURED, S.DATA_PREPARED, S.TRAINING, S.EVALUATING, S.MEASURING, S.COMPARING, S.REPORTING, S.IDLE]


def config():
    return parse_config(CONFIG)


def fake_point(cfg):
    ann = OpReport("ann", macs=100)
    snn = OpReport("snn", sops=60.0, state_updates=20.0, spikes_per_layer=[8.0, 3.0, 1.0])
    return PointResult(
        "demo_T4", "morgan", 10, 4, 0.7, 0.6, 0.65, 0.55, ann, snn,
        EfficiencyMetrics.compute(ann, snn, cfg.architecture, 4, [10, 5, 2]), cfg.architecture,
    )  # fmt: skip


class FakeRunner(ExperimentRunner):
    """Every stage is replaced by a recorder; `fail_at` makes one of them raise."""

    def __init__(self, fail_at=None):
        super().__init__(config(), "unused")
        self.calls, self.fail_at, self.released = [], fail_at, 0

    def _stage(self, name):
        self.calls.append(name)
        if name == self.fail_at:
            raise ValueError(f"boom in {name}")

    def _release(self):
        self.released += 1

    def _prepare_data(self):
        self._stage("_prepare_data")

    def _train(self):
        self._stage("_train")

    def _evaluate(self):
        self._stage("_evaluate")

    def _measure(self):
        self._stage("_measure")

    def _compare(self):
        self._stage("_compare")
        return fake_point(self.config)

    def _report(self, point):
        self._stage("_report")
        return {"csv": Path("metrics.csv")}


def test_run_goes_through_every_stage_in_order():
    runner = FakeRunner()
    result = runner.run()
    assert runner.calls == STAGES
    assert result.states == HAPPY_PATH
    assert result.point.point_id == "demo_T4" and result.paths == {"csv": Path("metrics.csv")}
    assert runner.state is S.IDLE and runner.released == 0


@pytest.mark.parametrize("stage, label", list(zip(STAGES, ["preparing data", "training", "evaluating", "measuring", "comparing", "reporting"])))
def test_a_failing_stage_is_reported_and_resources_released(stage, label):
    runner = FakeRunner(fail_at=stage)
    with pytest.raises(ExperimentError) as info:
        runner.run()
    assert info.value.stage == label
    assert isinstance(info.value.__cause__, ValueError) and "boom" in str(info.value)
    assert runner.state is S.IDLE and runner.released == 1
    assert runner.history[-2:] == [S.ERROR, S.IDLE]
    assert runner.calls[-1] == stage  # nothing ran after the failure


def test_a_runner_can_run_again_after_a_failure():
    runner = FakeRunner(fail_at="_train")
    with pytest.raises(ExperimentError):
        runner.run()
    runner.fail_at = None
    assert runner.run().states == HAPPY_PATH


def test_the_error_is_logged_with_its_traceback(caplog):
    with pytest.raises(ExperimentError), caplog.at_level("ERROR"):
        FakeRunner(fail_at="_measure").run()
    assert any("measuring failed" in r.getMessage() and r.exc_info for r in caplog.records)


def test_invalid_transitions_are_rejected():
    runner = FakeRunner()
    with pytest.raises(RuntimeError, match="idle -> training"):
        runner._transition(S.TRAINING)
    assert runner.state is S.IDLE


def test_every_working_state_can_fail_and_error_only_leads_to_idle():
    working = set(S) - {S.IDLE, S.ERROR}
    assert all(S.ERROR in TRANSITIONS[state] for state in working)
    assert TRANSITIONS[S.ERROR] == {S.IDLE}
    assert set(TRANSITIONS) == set(S)


def test_default_stages_call_the_pipeline_functions(monkeypatch, tmp_path):
    cfg, calls = config(), []
    point = fake_point(cfg)
    monkeypatch.setattr(runner_module, "prepare_data", lambda c: calls.append("prepare") or "prepared")
    monkeypatch.setattr(runner_module, "save_prepared", lambda prepared, path: calls.append(("save", prepared, path)))
    monkeypatch.setattr(runner_module, "train_model", lambda c, r, kind: calls.append(("train", kind)))
    monkeypatch.setattr(runner_module, "profile_models", lambda c, r: calls.append("profile"))
    monkeypatch.setattr(runner_module, "load_point_result", lambda c, r: calls.append("load") or point)
    monkeypatch.setattr(
        runner_module.ReportBuilder, "build_all", staticmethod(lambda results, d: calls.append(("build", results, d)) or {"csv": d})
    )
    result = ExperimentRunner(cfg, tmp_path).run()
    assert calls[0] == "prepare" and calls[1][:2] == ("save", "prepared")
    assert calls[1][2] == tmp_path / "demo" / "data" / "morgan1024_r2_s42"
    assert calls[2:4] == [("train", "ann"), ("train", "snn")]
    assert calls[4:7] == ["load", "profile", "load"]
    report_dir = tmp_path / "demo" / "morgan1024_r2_h8_T4_s42" / "report"
    assert calls[7] == ("build", [point], report_dir) and result.paths == {"csv": report_dir}
    assert result.states == HAPPY_PATH


# load_point_result -------------------------------------------------------------------------------


def write_point(cfg, tmp_path, ann_topology=(1024, 8, 2), snn_topology=(1024, 8, 2), skip=()):
    out = ResultsLayout(tmp_path, cfg).point_dir
    out.mkdir(parents=True)
    for kind, topology, auc, bal in (("ann", ann_topology, 0.74, 0.69), ("snn", snn_topology, 0.70, 0.66)):
        metrics = {"topology": list(topology), "test": {"roc_auc": auc, "balanced_accuracy": bal}}
        if kind == "snn":
            metrics["snn"] = {"T": 4}
        files = {f"{kind}.pt": b"", f"{kind}_metrics.json": json.dumps(metrics).encode()}
        for name, data in files.items():
            if name not in skip:
                (out / name).write_bytes(data)
    OpReport("ann", macs=1024 * 8 + 8 * 2).save(out / "opreport_ann.json")
    OpReport("snn", sops=500.0, state_updates=40.0, spikes_per_layer=[27.0, 3.0, 1.0]).save(out / "opreport_snn.json")


def test_load_point_result(tmp_path):
    cfg = config()
    write_point(cfg, tmp_path)
    p = load_point_result(cfg, tmp_path)
    assert (p.point_id, p.fingerprint, p.n_bits, p.T) == ("morgan1024_r2_h8_T4_s42", "morgan", 1024, 4)
    assert (p.ann_auc, p.snn_auc, p.ann_balanced_accuracy, p.snn_balanced_accuracy) == (0.74, 0.70, 0.69, 0.66)
    assert p.metrics.n_mac == 1024 * 8 + 16 and p.metrics.n_sop == 500.0
    assert p.metrics.r == pytest.approx(500 / (4 * p.metrics.n_mac))
    assert p.metrics.rate_per_layer == pytest.approx([27 / (1024 * 4), 3 / (8 * 4), 1 / (2 * 4)])
    assert p.params == cfg.architecture and p.group == "morgan1024_h8"  # no seed in the hand-written metrics


def test_load_point_result_uses_the_real_input_size_for_maccs(tmp_path):
    raw = json.loads(json.dumps(CONFIG))
    raw["data"]["fingerprint"] = {"type": "maccs", "n_bits": 1024, "radius": 2}
    cfg = parse_config(raw)
    write_point(cfg, tmp_path, (166, 8, 2), (166, 8, 2))
    OpReport("ann", macs=166 * 8 + 16).save(ResultsLayout(tmp_path, cfg).point_dir / "opreport_ann.json")
    OpReport("snn", sops=50.0, state_updates=40.0, spikes_per_layer=[10.0, 3.0, 1.0]).save(ResultsLayout(tmp_path, cfg).point_dir / "opreport_snn.json")
    assert load_point_result(cfg, tmp_path).group == "maccs166_h8"


@pytest.mark.parametrize("missing", ["snn.pt", "ann_metrics.json"])
def test_load_point_result_needs_both_models(tmp_path, missing):
    cfg = config()
    write_point(cfg, tmp_path, skip=(missing,))
    with pytest.raises(FileNotFoundError, match=missing.replace(".", r"\.")):
        load_point_result(cfg, tmp_path)


def test_load_point_result_rejects_non_homologous_models(tmp_path):
    cfg = config()
    write_point(cfg, tmp_path, snn_topology=(1024, 16, 2))
    with pytest.raises(ValueError, match="homologous"):
        load_point_result(cfg, tmp_path)
