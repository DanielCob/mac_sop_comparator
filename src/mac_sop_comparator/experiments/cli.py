"""Command-line interface: `python -m mac_sop_comparator <command> --config <file.yaml>`."""

import argparse
import logging
import sys
from pathlib import Path

from mac_sop_comparator.config import Config, ConfigError, load_config
from mac_sop_comparator.data import DataError, prepare_data, prepared_dir, save_prepared
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
    setup_logging(results_dir / config.experiment.name / "run.log")
    set_seed(config.experiment.seed)
    n_threads = set_num_threads(config.experiment.n_threads)
    logger.info("experiment '%s': seed %d, %d CPU threads", config.experiment.name, config.experiment.seed, n_threads)


def _prepare(config: Config, results_dir: Path) -> int:
    logger.info("[1/2] preparing data")
    prepared = prepare_data(config)
    logger.info("[2/2] saving")
    out = save_prepared(prepared, prepared_dir(results_dir, config))
    s = prepared.summary
    print(
        f"prepared {s['n_molecules']} molecules ({s['n_features']} features, "
        f"{100 * s['positive_rate']:.1f}% positive, sparsity {s['sparsity']:.3f}) -> {out}"
    )
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
    except DataError as exc:
        logger.error("%s", exc)
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    print(f"error: '{args.command}' is not implemented yet", file=sys.stderr)
    return EXIT_ERROR
