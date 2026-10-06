"""Command-line interface: `python -m mac_sop_comparator <command> --config <file.yaml>`."""

import argparse
import sys

COMMANDS = {
    "prepare": "fingerprints, labels and splits",
    "train": "train and evaluate a model (--model ann|snn)",
    "profile": "measure the complexity of the trained models",
    "compare": "unified metrics, tables and plots",
    "sweep": "design-space sweep",
    "run": "prepare + train + profile + compare",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mac_sop_comparator",
        description="Efficiency comparison of a dense ANN (MACs) vs a LIF SNN (SOPs).",
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="command")
    for name, help_text in COMMANDS.items():
        cmd = sub.add_parser(name, help=help_text, description=help_text)
        cmd.add_argument("--config", required=True, help="path to the experiment YAML")
        if name == "train":
            cmd.add_argument("--model", required=True, choices=["ann", "snn"])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(f"'{args.command}' is not implemented yet", file=sys.stderr)
    return 1
