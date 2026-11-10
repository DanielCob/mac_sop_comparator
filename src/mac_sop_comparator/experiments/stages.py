"""Pipeline stages shared by the CLI commands and the ExperimentRunner."""

import hashlib
import json
import logging
import time
from pathlib import Path

from mac_sop_comparator.comparison import EfficiencyMetrics, PointResult
from mac_sop_comparator.config import Config
from mac_sop_comparator.data import load_prepared
from mac_sop_comparator.experiments.layout import KINDS, ResultsLayout
from mac_sop_comparator.models import ModelFactory, SnnClassifier, load_checkpoint, save_checkpoint
from mac_sop_comparator.profiling import MacProfiler, OpReport, SopProfiler, TrafficModel
from mac_sop_comparator.training import Evaluator, Trainer, plot_training_curve
from mac_sop_comparator.utils import get_device, set_seed

logger = logging.getLogger(__name__)


def _profile(model, test_data) -> OpReport:
    """MACs for the ANN, SOPs for the SNN, over the whole test set (D-31)."""
    profiler = SopProfiler if isinstance(model, SnnClassifier) else MacProfiler
    return profiler(model).measure(test_data)


def ann_cache_key(config: Config) -> str:
    """Hash of everything the trained ANN depends on (not T, beta or the threshold): the sweep reuses it (D-26, D-44)."""
    payload = {
        "data": config.data.model_dump(),
        "hidden_layers": config.model.hidden_layers,
        "n_outputs": config.model.n_outputs,
        "ann": config.model.ann.model_dump(),
        "training": config.training.model_dump(),
        "seed": config.experiment.seed,
        "device": config.experiment.device,
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def point_key(config: Config) -> str:
    """Hash of everything a trained point depends on (data, model with T, training, seed, device), not the energies.

    The SNN saves it (the ANN has its own `cache_key`, which does not depend on T) so that the sweep never reuses
    results trained under another configuration (D-48).
    """
    payload = {
        "data": config.data.model_dump(),
        "model": config.model.model_dump(),
        "training": config.training.model_dump(),
        "seed": config.experiment.seed,
        "device": config.experiment.device,
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def train_model(config: Config, results_dir: str | Path, kind: str) -> dict:
    """Train and evaluate one model on the prepared data; save checkpoint, metrics, curve and OpReport."""
    layout = ResultsLayout(results_dir, config)
    layout.model_path(kind)  # validates the kind
    prepared = load_prepared(layout.data_dir)
    parts = prepared.partitions()
    device = get_device(config.experiment.device)

    set_seed(config.experiment.seed)  # the ANN and the SNN start from the same initial weights
    model = ModelFactory.build(config.model, prepared.X.shape[1], kind)
    logger.info("training the %s %s on %s", kind.upper(), model.topology, device)
    started = time.perf_counter()
    result = Trainer.from_config(config.training, device, config.experiment.seed).train(model, parts["train"], parts["val"])
    seconds = time.perf_counter() - started

    evaluator = Evaluator()
    metrics = {
        "model": kind,
        "topology": model.topology,
        "seed": config.experiment.seed,
        "device": str(device),
        "best_epoch": result.best_epoch,
        "epochs_run": len(result.history),
        "stopped_early": result.stopped_early,
        "validation": evaluator.evaluate(model, parts["val"]),
        "test": evaluator.evaluate(model, parts["test"]),
        "train_seconds": round(seconds, 2),  # wall-clock, informative only: not a comparison metric
        "history": result.history,
    }
    if kind == "ann":
        metrics["cache_key"] = ann_cache_key(config)
    if kind == "snn":
        metrics["config_key"] = point_key(config)
        metrics["snn"] = {"T": model.T, "beta": model.beta, "threshold": model.threshold, "reset": model.reset}
    save_checkpoint(model, layout.model_path(kind))
    plot_training_curve(result.history, result.best_epoch, layout.curve_path(kind), f"{config.experiment.name} - {kind.upper()}")
    layout.metrics_path(kind).write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    report = _profile(model, parts["test"])
    report.save(layout.opreport_path(kind))
    logger.info("test: %s", metrics["test"])
    return {"directory": layout.point_dir, "metrics": metrics, "report": report}


def profile_models(config: Config, results_dir: str | Path) -> dict[str, OpReport]:
    """Measure the operations of the trained models found for this point (ANN and/or SNN)."""
    layout = ResultsLayout(results_dir, config)
    found = [kind for kind in KINDS if layout.model_path(kind).is_file()]
    if not found:
        raise FileNotFoundError(f"no trained model in {layout.point_dir}; run the 'train' command first (--model ann or snn)")
    test = load_prepared(layout.data_dir).partition("test")
    reports = {}
    for kind in found:
        reports[kind] = _profile(load_checkpoint(layout.model_path(kind)), test)
        reports[kind].save(layout.opreport_path(kind))
    return reports


def load_point_result(config: Config, results_dir: str | Path) -> PointResult:
    """Rebuild the PointResult of a point from the artifacts of its trained ANN and SNN (D-39)."""
    layout = ResultsLayout(results_dir, config)
    needed = [path for kind in KINDS for path in (layout.model_path(kind), layout.metrics_path(kind), layout.opreport_path(kind))]
    missing = [path.name for path in needed if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{layout.point_dir} lacks {missing}; train and profile both models first")
    metrics = {kind: json.loads(layout.metrics_path(kind).read_text(encoding="utf-8")) for kind in KINDS}
    ann, snn = (OpReport.load(layout.opreport_path(kind)) for kind in KINDS)
    if metrics["ann"]["topology"] != metrics["snn"]["topology"]:
        raise ValueError(f"the ANN {metrics['ann']['topology']} and the SNN {metrics['snn']['topology']} are not homologous")
    topology = metrics["ann"]["topology"]
    T = metrics["snn"]["snn"]["T"]
    fp = config.data.fingerprint
    traffic = TrafficModel(config.architecture, topology)  # bytes depend on the parameters, so they are not stored (D-42)
    return PointResult(
        point_id=layout.point_id,
        fingerprint=fp.type,
        n_bits=topology[0],  # the real input size: 166 for MACCS whatever `n_bits` says
        T=T,
        ann_auc=metrics["ann"]["test"]["roc_auc"],
        snn_auc=metrics["snn"]["test"]["roc_auc"],
        ann_balanced_accuracy=metrics["ann"]["test"]["balanced_accuracy"],
        snn_balanced_accuracy=metrics["snn"]["test"]["balanced_accuracy"],
        ann_report=traffic.annotate(ann),
        snn_report=traffic.annotate(snn),
        metrics=EfficiencyMetrics.compute(ann, snn, config.architecture, T, topology),
        params=config.architecture,
        hidden_layers=tuple(topology[1:-1]),
        seed=metrics["ann"].get("seed"),
    )
