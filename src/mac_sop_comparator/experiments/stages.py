"""Pipeline stages shared by the CLI commands (and later by the ExperimentRunner)."""

import json
import logging
import time
from pathlib import Path

from mac_sop_comparator.config import Config, FingerprintConfig
from mac_sop_comparator.data import fingerprint_id, load_prepared, prepared_dir
from mac_sop_comparator.models import ModelFactory, SnnClassifier, load_checkpoint, save_checkpoint
from mac_sop_comparator.profiling import MacProfiler, OpReport, SopProfiler
from mac_sop_comparator.training import Evaluator, Trainer, plot_training_curve
from mac_sop_comparator.utils import get_device, set_seed

logger = logging.getLogger(__name__)

KINDS = ("ann", "snn")


def point_id(config: Config, fp: FingerprintConfig | None = None) -> str:
    """Identifies a design-space point: fingerprint and SNN time steps, e.g. `morgan1024_r2_T10` (D-25)."""
    return f"{fingerprint_id(fp or config.data.fingerprint)}_T{config.model.snn.T}"


def point_dir(results_dir: str | Path, config: Config, fp: FingerprintConfig | None = None) -> Path:
    return Path(results_dir) / config.experiment.name / point_id(config, fp)


def _profile(model, test_data) -> OpReport:
    """MACs for the ANN, SOPs for the SNN, over the whole test set (D-31)."""
    profiler = SopProfiler if isinstance(model, SnnClassifier) else MacProfiler
    return profiler(model).measure(test_data)


def train_model(config: Config, results_dir: str | Path, kind: str) -> dict:
    """Train and evaluate one model on the prepared data; save checkpoint, metrics, curve and OpReport."""
    if kind not in KINDS:
        raise ValueError(f"model kind must be one of {KINDS}, got '{kind}'")
    prepared = load_prepared(prepared_dir(results_dir, config))
    parts = prepared.partitions()
    out = point_dir(results_dir, config)
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
    if kind == "snn":
        metrics["snn"] = {"T": model.T, "beta": model.beta, "threshold": model.threshold, "reset": model.reset}
    save_checkpoint(model, out / f"{kind}.pt")
    plot_training_curve(result.history, result.best_epoch, out / f"{kind}_curve.png", f"{config.experiment.name} - {kind.upper()}")
    (out / f"{kind}_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    report = _profile(model, parts["test"])
    report.save(out / f"opreport_{kind}.json")
    logger.info("test: %s", metrics["test"])
    return {"directory": out, "metrics": metrics, "report": report}


def profile_models(config: Config, results_dir: str | Path) -> dict[str, OpReport]:
    """Measure the operations of the trained models found for this point (ANN and/or SNN)."""
    out = point_dir(results_dir, config)
    found = [kind for kind in KINDS if (out / f"{kind}.pt").is_file()]
    if not found:
        raise FileNotFoundError(f"no trained model in {out}; run the 'train' command first (--model ann or snn)")
    test = load_prepared(prepared_dir(results_dir, config)).partition("test")
    reports = {}
    for kind in found:
        reports[kind] = _profile(load_checkpoint(out / f"{kind}.pt"), test)
        reports[kind].save(out / f"opreport_{kind}.json")
    return reports
