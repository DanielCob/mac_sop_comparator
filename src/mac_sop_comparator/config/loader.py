"""Load and validate an experiment YAML into a `Config`."""

from pathlib import Path

import yaml
from pydantic import ValidationError

from mac_sop_comparator.config.schema import Config


class ConfigError(ValueError):
    """The configuration file is missing, unreadable or invalid."""


def load_config(path: str | Path) -> Config:
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: invalid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: the top level must be a mapping of sections")
    return parse_config(raw, source=str(path))


def parse_config(raw: dict, source: str = "<dict>") -> Config:
    try:
        return Config.model_validate(raw)
    except ValidationError as exc:
        lines = [f"  {'.'.join(str(p) for p in e['loc']) or '<root>'}: {e['msg']}" for e in exc.errors()]
        raise ConfigError(f"{source}: invalid configuration\n" + "\n".join(lines)) from exc
