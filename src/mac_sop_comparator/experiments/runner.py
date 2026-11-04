"""ExperimentRunner: one design-space point from data to report, as a state machine (Template Method, D-39)."""

import gc
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import torch

from mac_sop_comparator.comparison import PointResult, ReportBuilder
from mac_sop_comparator.config import Config
from mac_sop_comparator.data import prepare_data, save_prepared
from mac_sop_comparator.experiments.layout import KINDS, ResultsLayout
from mac_sop_comparator.experiments.stages import load_point_result, profile_models, train_model

logger = logging.getLogger(__name__)


class RunnerState(Enum):
    IDLE = "idle"
    CONFIGURED = "configured"
    DATA_PREPARED = "data_prepared"
    TRAINING = "training"
    EVALUATING = "evaluating"
    MEASURING = "measuring"
    COMPARING = "comparing"
    REPORTING = "reporting"
    ERROR = "error"


S = RunnerState
TRANSITIONS = {
    S.IDLE: {S.CONFIGURED},
    S.CONFIGURED: {S.DATA_PREPARED, S.ERROR},
    S.DATA_PREPARED: {S.TRAINING, S.ERROR},
    S.TRAINING: {S.EVALUATING, S.ERROR},
    S.EVALUATING: {S.MEASURING, S.ERROR},
    S.MEASURING: {S.COMPARING, S.ERROR},
    S.COMPARING: {S.REPORTING, S.ERROR},
    S.REPORTING: {S.IDLE, S.ERROR},
    S.ERROR: {S.IDLE},
}


class ExperimentError(Exception):
    """A stage of the run failed; `stage` says which and the original exception is the `__cause__`."""

    def __init__(self, stage: str, cause: Exception):
        super().__init__(f"{stage} failed: {cause}")
        self.stage = stage


@dataclass(frozen=True)
class RunResult:
    point: PointResult
    paths: dict[str, Path]  # CSV, JSON and PNG reports
    states: list[RunnerState]  # every state visited, in order


class ExperimentRunner:
    """Runs one point: prepare data, train the ANN and the SNN, evaluate, measure, compare and report.

    `run()` is the template; each stage is a method a subclass may replace (the tests inject fake stages).
    The stages recompute the point, with no hidden cache (D-39). The sweep asks explicitly for less work (D-44):
    `kinds` are the models to train (the ANN may have been copied from another T) and `reuse_data` keeps the
    prepared data already on disk.
    """

    def __init__(
        self, config: Config, results_dir: str | Path = "results", kinds: tuple[str, ...] = KINDS, reuse_data: bool = False
    ):
        self.config = config
        self.results_dir = Path(results_dir)
        self.kinds = kinds
        self.reuse_data = reuse_data
        self.layout = ResultsLayout(results_dir, config)
        self.state = S.IDLE
        self.history: list[RunnerState] = [S.IDLE]

    def _transition(self, new: RunnerState) -> None:
        if new not in TRANSITIONS[self.state]:
            raise RuntimeError(f"invalid transition {self.state.value} -> {new.value}")
        logger.info("state: %s -> %s", self.state.value, new.value)
        self.state = new
        self.history.append(new)

    def _release(self) -> None:
        """Free what a failed stage may hold: Python garbage and the MPS cache."""
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    def run(self) -> RunResult:
        self.history = [self.state]
        stage = "configuring"
        try:
            self._transition(S.CONFIGURED)
            stage = "preparing data"
            self._prepare_data()
            self._transition(S.DATA_PREPARED)
            stage = "training"
            self._transition(S.TRAINING)
            self._train()
            stage = "evaluating"
            self._transition(S.EVALUATING)
            self._evaluate()
            stage = "measuring"
            self._transition(S.MEASURING)
            self._measure()
            stage = "comparing"
            self._transition(S.COMPARING)
            point = self._compare()
            stage = "reporting"
            self._transition(S.REPORTING)
            paths = self._report(point)
            self._transition(S.IDLE)
        except Exception as exc:
            logger.exception("%s failed", stage)
            logger.info("state: %s -> %s", self.state.value, S.ERROR.value)
            self.state = S.ERROR  # reachable from every working state
            self.history.append(S.ERROR)
            self._release()
            self._transition(S.IDLE)
            raise ExperimentError(stage, exc) from exc
        return RunResult(point=point, paths=paths, states=list(self.history))

    # Stages ---------------------------------------------------------------------------------------

    def _prepare_data(self) -> None:
        if self.reuse_data and (self.layout.data_dir / "dataset.npz").is_file():
            logger.info("reusing the prepared data in %s", self.layout.data_dir)
            return
        save_prepared(prepare_data(self.config), self.layout.data_dir)

    def _train(self) -> None:
        """Trains the models in `kinds`; `train_model` also scores them on validation and test and saves their curves."""
        for kind in self.kinds:
            train_model(self.config, self.results_dir, kind)

    def _evaluate(self) -> None:
        """The test-set scores were computed by the training stage; this checks that both exist on disk."""
        load_point_result(self.config, self.results_dir)

    def _measure(self) -> None:
        """Measures the operations of the saved checkpoints (the same path as the `profile` command)."""
        profile_models(self.config, self.results_dir)

    def _compare(self) -> PointResult:
        return load_point_result(self.config, self.results_dir)

    def _report(self, point: PointResult) -> dict[str, Path]:
        return ReportBuilder.build_all([point], self.layout.report_dir)
