"""Command-line interface: `python -m mac_sop_comparator <command> --config <file.yaml>`."""

import argparse
import logging
import sys
from pathlib import Path

from mac_sop_comparator.comparison import PointResult, ReportBuilder
from mac_sop_comparator.config import Config, ConfigError, load_config
from mac_sop_comparator.data import DataError, prepare_data, save_prepared
from mac_sop_comparator.experiments.layout import ResultsLayout
from mac_sop_comparator.experiments.runner import ExperimentError, ExperimentRunner
from mac_sop_comparator.experiments.stages import load_point_result, profile_models, train_model
from mac_sop_comparator.utils import set_num_threads, set_seed, setup_logging

logger = logging.getLogger(__name__)

COMMANDS = {
    "prepare": "fingerprints, labels and splits",
    "train": "train and evaluate a model (--model ann|snn)",
    "profile": "measure the complexity of the trained models",
    "compare": "unified metrics, tables and plots",
    "sweep": "design-space sweep",
    "run": "prepare + train + profile + compare",
}

EXIT_OK, EXIT_ERROR, EXIT_CONFIG_ERROR = 0, 1, 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mac_sop_comparator",
        description="Efficiency comparison of a dense ANN (MACs) vs a LIF SNN (SOPs).",
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="command")
    for name, help_text in COMMANDS.items():
        cmd = sub.add_parser(name, help=help_text, description=help_text)
        cmd.add_argument("--config", required=True, help="path to the experiment YAML")
        cmd.add_argument("--results-dir", default="results", help="where artifacts are written (default: results)")
        if name == "train":
            cmd.add_argument("--model", required=True, choices=["ann", "snn"])
    return parser


def _setup(config: Config, results_dir: Path) -> None:
    setup_logging(ResultsLayout(results_dir, config).log_path)
    set_seed(config.experiment.seed)
    n_threads = set_num_threads(config.experiment.n_threads)
    logger.info("experiment '%s': seed %d, %d CPU threads", config.experiment.name, config.experiment.seed, n_threads)


def _prepare(config: Config, results_dir: Path) -> int:
    logger.info("[1/2] preparing data")
    prepared = prepare_data(config)
    logger.info("[2/2] saving")
    out = save_prepared(prepared, ResultsLayout(results_dir, config).data_dir)
    s = prepared.summary
    print(
        f"prepared {s['n_molecules']} molecules ({s['n_features']} features, "
        f"{100 * s['positive_rate']:.1f}% positive, sparsity {s['sparsity']:.3f}) -> {out}"
    )
    return EXIT_OK


def _train(config: Config, results_dir: Path, kind: str) -> int:
    out = train_model(config, results_dir, kind)
    m, report = out["metrics"], out["report"]
    cost = f"{report.macs} MACs" if kind == "ann" else f"{report.sops:.0f} SOPs"
    print(
        f"trained the {kind.upper()} {m['topology']} for {m['epochs_run']} epochs (best {m['best_epoch']}): "
        f"validation ROC-AUC {m['validation']['roc_auc']:.4f}, test ROC-AUC {m['test']['roc_auc']:.4f}, "
        f"test balanced accuracy {m['test']['balanced_accuracy']:.4f}, {cost} per inference -> {out['directory']}"
    )
    return EXIT_OK


def _profile(config: Config, results_dir: Path) -> int:
    for name, report in profile_models(config, results_dir).items():
        if name == "ann":
            print(f"ann: {report.macs} MACs, {report.biases} biases (per inference)")
        else:
            print(
                f"snn: {report.sops:.1f} SOPs, {report.state_updates:.0f} state updates, {report.biases} biases "
                f"(per inference, mean over the test set)"
            )
    return EXIT_OK


def _summary(point: PointResult, report_dir: Path) -> str:
    m, p = point.metrics, point.params
    verdict = "SNN cheaper" if m.snn_cheaper else "ANN cheaper"
    missing = p.missing_sources()
    sources = "every parameter cites its source" if not missing else f"WARNING: no source for {', '.join(missing)}"
    return "\n".join(
        [
            f"point {point.point_id} (T = {point.T})",
            f"ROC-AUC (test)     ANN {point.ann_auc:.4f}   SNN {point.snn_auc:.4f}",
            f"operations         ANN {m.n_mac:.0f} MACs   SNN {m.n_sop:.0f} SOPs + {m.n_upd:.0f} state updates",
            f"activity           r = {m.r:.4f}   critical r* = {m.r_critical:.4f}",
            f"relative cost      ANN {m.cost_ann:.0f}   SNN {m.cost_snn:.0f}   (ratio {m.cost_ratio:.4f}, {verdict})",
            f"memory traffic     ANN {m.bytes_ann / 1024:.1f} KB (von Neumann, AI {m.ai_ann:.2f})   "
            f"SNN {m.bytes_snn / 1024:.2f} KB (neuromorphic, AI {m.ai_snn:.1f})",
            f"total energy       ANN {m.energy_ann:.0f}   SNN {m.energy_snn:.0f}   (ratio {m.energy_ratio:.6f})",
            f"costs use the energies of the config (e_mac {p.e_mac:g}, e_ac {p.e_ac:g}, e_upd {p.e_upd:g})",
            f"sources            {sources}",
            f"reports -> {report_dir}",
        ]
    )


def _compare(config: Config, results_dir: Path) -> int:
    layout = ResultsLayout(results_dir, config)
    point = load_point_result(config, results_dir)
    ReportBuilder.build_all([point], layout.report_dir)
    print(_summary(point, layout.report_dir))
    return EXIT_OK


def _run(config: Config, results_dir: Path) -> int:
    result = ExperimentRunner(config, results_dir).run()
    print(_summary(result.point, ResultsLayout(results_dir, config).report_dir))
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_CONFIG_ERROR
    results_dir = Path(args.results_dir)
    _setup(config, results_dir)
    try:
        if args.command == "prepare":
            return _prepare(config, results_dir)
        if args.command == "train":
            return _train(config, results_dir, args.model)
        if args.command == "profile":
            return _profile(config, results_dir)
        if args.command == "compare":
            return _compare(config, results_dir)
        if args.command == "run":
            return _run(config, results_dir)
    except (DataError, FileNotFoundError, ExperimentError) as exc:
        logger.error("%s", exc)
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    print(f"error: '{args.command}' is not implemented yet", file=sys.stderr)
    return EXIT_ERROR
