"""Pipeline stages shared by the CLI commands (and later by the ExperimentRunner)."""

import json
import logging
import time
from pathlib import Path

from mac_sop_comparator.config import Config, FingerprintConfig
from mac_sop_comparator.data import fingerprint_id, load_prepared, prepared_dir
from mac_sop_comparator.models import ModelFactory, load_checkpoint, save_checkpoint
from mac_sop_comparator.profiling import MacProfiler, OpReport
from mac_sop_comparator.training import Evaluator, Trainer
from mac_sop_comparator.utils import get_device, set_seed

logger = logging.getLogger(__name__)


def point_id(config: Config, fp: FingerprintConfig | None = None) -> str:
    """Identifies a design-space point: fingerprint and SNN time steps, e.g. `morgan1024_r2_T10` (D-25)."""
    return f"{fingerprint_id(fp or config.data.fingerprint)}_T{config.model.snn.T}"


def point_dir(results_dir: str | Path, config: Config, fp: FingerprintConfig | None = None) -> Path:
    return Path(results_dir) / config.experiment.name / point_id(config, fp)


def train_ann(config: Config, results_dir: str | Path) -> dict:
    """Train and evaluate the ANN on the prepared data; save checkpoint, metrics and OpReport."""
    prepared = load_prepared(prepared_dir(results_dir, config))
    parts = prepared.partitions()
    out = point_dir(results_dir, config)
    device = get_device(config.experiment.device)

    set_seed(config.experiment.seed)  # same weights for the same seed, whatever ran before
    model = ModelFactory.build(config.model, prepared.X.shape[1], "ann")
    logger.info("training the ANN %s on %s", model.topology, device)
    started = time.perf_counter()
    result = Trainer.from_config(config.training, device, config.experiment.seed).train(model, parts["train"], parts["val"])
    seconds = time.perf_counter() - started

    evaluator = Evaluator()
    metrics = {
        "model": "ann",
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
    save_checkpoint(model, out / "ann.pt")
    (out / "ann_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    report = MacProfiler(model).measure(parts["test"])
    report.save(out / "opreport_ann.json")
    logger.info("test: %s", metrics["test"])
    return {"directory": out, "metrics": metrics, "report": report}


def profile_models(config: Config, results_dir: str | Path) -> dict[str, OpReport]:
    """Measure the operations of the trained models found for this point (the ANN so far)."""
    out = point_dir(results_dir, config)
    checkpoint = out / "ann.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"no trained ANN in {out}; run the 'train --model ann' command first")
    model = load_checkpoint(checkpoint)
    test = load_prepared(prepared_dir(results_dir, config)).partition("test")  # D-31: the whole test set
    report = MacProfiler(model).measure(test)
    report.save(out / "opreport_ann.json")
    return {"ann": report}
