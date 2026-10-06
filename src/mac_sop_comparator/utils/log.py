"""Logging to console and to a .log file, including library versions."""

import logging
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

PACKAGES = ("torch", "snntorch", "rdkit", "numpy", "pandas", "scikit-learn", "PyYAML", "matplotlib")
FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def setup_logging(log_file: str | Path | None = None, level: int = logging.INFO) -> None:
    """Configure the root logger (console, plus a file if given) and log the environment."""
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(level=level, format=FORMAT, handlers=handlers, force=True)
    log_environment()


def log_environment() -> None:
    logger = logging.getLogger(__name__)
    logger.info("Python %s on %s", platform.python_version(), platform.platform())
    for name in PACKAGES:
        try:
            logger.info("%s %s", name, version(name))
        except PackageNotFoundError:
            logger.warning("%s is not installed", name)
