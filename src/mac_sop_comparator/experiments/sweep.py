"""SweepManager: the design-space sweep, one ExperimentRunner per point (D-26, D-36, D-44)."""

import csv
import itertools
import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

from mac_sop_comparator.comparison import PointResult, ReportBuilder
from mac_sop_comparator.config import Config, FingerprintConfig
from mac_sop_comparator.experiments.layout import ResultsLayout
from mac_sop_comparator.experiments.runner import ExperimentError, ExperimentRunner
from mac_sop_comparator.experiments.stages import ann_cache_key, load_point_result, point_key

logger = logging.getLogger(__name__)

ANN = "ann"


@dataclass(frozen=True)
class SweepResult:
    points: list[PointResult]  # every point that has a result, in sweep order (run now or found on disk)
    failed: dict[str, str]  # point id -> error
    skipped: list[str]  # ids whose report already existed
    reused_ann: list[str]  # ids whose ANN was copied from another T instead of trained
    paths: dict[str, Path]  # consolidated files


class SweepManager:
    """Cartesian product of the `sweep` section over seeds, fingerprints, topologies and T.

    Points are ordered so that those sharing an ANN are consecutive: the first one trains it and the rest copy it
    (D-26). A failed point is recorded and the sweep goes on; a point whose report exists is skipped unless `force`.
    """

    def __init__(self, config: Config, results_dir: str | Path = "results"):
        if config.sweep is None:
            raise ValueError("the config has no 'sweep' section")
        self.config = config
        self.results_dir = Path(results_dir)

    # Points ---------------------------------------------------------------------------------------

    def _fingerprints(self) -> list[FingerprintConfig]:
        base = self.config.data.fingerprint
        sweep = self.config.sweep
        fingerprints = []
        for kind in dict.fromkeys(sweep.fingerprint):
            if kind == "maccs":
                fingerprints.append(base.model_copy(update={"type": "maccs"}))
            else:
                for n_bits in dict.fromkeys(sweep.n_bits or [base.n_bits]):
                    fingerprints.append(base.model_copy(update={"type": "morgan", "n_bits": n_bits}))
        return fingerprints

    def configs(self) -> list[Config]:
        """The config of every point; more than `max_points` is an error, never a silent cut."""
        sweep, base = self.config.sweep, self.config
        seeds = list(dict.fromkeys(sweep.seeds or [base.experiment.seed]))
        topologies = [list(h) for h in (sweep.hidden_layers or [base.model.hidden_layers])]
        configs = {}
        for seed, fp, hidden, T in itertools.product(seeds, self._fingerprints(), topologies, dict.fromkeys(sweep.T)):
            point = base.model_copy(
                update={
                    "experiment": base.experiment.model_copy(update={"seed": seed}),
                    "data": base.data.model_copy(update={"fingerprint": fp}),
                    "model": base.model.model_copy(
                        update={"hidden_layers": hidden, "snn": base.model.snn.model_copy(update={"T": T})}
                    ),
                }
            )
            configs.setdefault(ResultsLayout(self.results_dir, point).point_id, point)
        if len(configs) > sweep.max_points:
            raise ValueError(f"the sweep has {len(configs)} points, more than max_points = {sweep.max_points}")
        return list(configs.values())

    # ANN reuse ------------------------------------------------------------------------------------

    def _ann_files(self, layout: ResultsLayout) -> list[Path]:
        return [layout.model_path(ANN), layout.metrics_path(ANN), layout.curve_path(ANN), layout.opreport_path(ANN)]

    def _cached_ann(self, config: Config, configs: list[Config]) -> ResultsLayout | None:
        """Another point of the sweep that already holds an ANN trained under the same cache key."""
        key = ann_cache_key(config)
        for other in configs:
            layout = ResultsLayout(self.results_dir, other)
            if layout.point_id == ResultsLayout(self.results_dir, config).point_id:
                continue
            files = self._ann_files(layout)
            if all(f.is_file() for f in files) and json.loads(files[1].read_text(encoding="utf-8")).get("cache_key") == key:
                return layout
        return None

    def _copy_ann(self, source: ResultsLayout, target: ResultsLayout) -> None:
        target.point_dir.mkdir(parents=True, exist_ok=True)
        for src, dst in zip(self._ann_files(source), self._ann_files(target)):
            shutil.copy2(src, dst)

    def _up_to_date(self, config: Config, layout: ResultsLayout) -> bool:
        """The saved models of the point were trained under this very configuration (old results lack the key)."""
        try:
            ann = json.loads(layout.metrics_path(ANN).read_text(encoding="utf-8"))
            snn = json.loads(layout.metrics_path("snn").read_text(encoding="utf-8"))
        except FileNotFoundError:
            return False
        return ann.get("cache_key") == ann_cache_key(config) and snn.get("config_key") == point_key(config)

    def collect(self) -> tuple[list[PointResult], list[str]]:
        """The results already on disk for the points of this sweep, and the ids that have none or are stale (D-50)."""
        points, missing = [], []
        for config in self.configs():
            layout = ResultsLayout(self.results_dir, config)
            if not self._up_to_date(config, layout):
                missing.append(layout.point_id)
                continue
            try:
                points.append(load_point_result(config, self.results_dir))
            except (FileNotFoundError, ValueError):
                missing.append(layout.point_id)
        return points, missing

    # Run ------------------------------------------------------------------------------------------

    def run(self, force: bool = False) -> SweepResult:
        configs = self.configs()
        points, failed, skipped, reused = [], {}, [], []
        for i, config in enumerate(configs, start=1):
            layout = ResultsLayout(self.results_dir, config)
            logger.info("sweep point %d/%d: %s", i, len(configs), layout.point_id)
            if not force and (layout.report_dir / "metrics.csv").is_file() and not self._up_to_date(config, layout):
                logger.warning("%s was trained with another configuration; running it again", layout.point_id)
            elif not force and (layout.report_dir / "metrics.csv").is_file():
                try:
                    point = load_point_result(config, self.results_dir)
                    ReportBuilder.build_all([point], layout.report_dir)  # refresh with the current parameters
                    points.append(point)
                    skipped.append(layout.point_id)
                    continue
                except (FileNotFoundError, ValueError) as exc:
                    logger.warning("%s has a report but cannot be reloaded (%s); running it again", layout.point_id, exc)
            kinds = (ANN, "snn")
            source = self._cached_ann(config, configs)
            if source is not None:
                logger.info("%s: copying the ANN of %s", layout.point_id, source.point_id)
                self._copy_ann(source, layout)
                kinds = ("snn",)
                reused.append(layout.point_id)
            try:
                points.append(ExperimentRunner(config, self.results_dir, kinds=kinds, reuse_data=True).run().point)
            except ExperimentError as exc:
                logger.error("%s: %s", layout.point_id, exc)
                failed[layout.point_id] = str(exc)
                if kinds == ("snn",):
                    reused.remove(layout.point_id)
        return SweepResult(points, failed, skipped, reused, self._consolidate(points, failed))

    def _consolidate(self, points: list[PointResult], failed: dict[str, str]) -> dict[str, Path]:
        sweep_dir = ResultsLayout(self.results_dir, self.config).sweep_dir
        paths = {}
        if points:
            paths.update(ReportBuilder.build_sweep(points, sweep_dir))
        failed_path = sweep_dir / "failed.csv"
        if failed:
            failed_path.parent.mkdir(parents=True, exist_ok=True)
            with failed_path.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["point_id", "error"])
                writer.writerows(failed.items())
            paths["failed"] = failed_path
        else:
            failed_path.unlink(missing_ok=True)  # a clean run leaves no stale failures
        return paths
