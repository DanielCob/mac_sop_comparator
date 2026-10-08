import copy

import pytest
import yaml
from pydantic import ValidationError

from mac_sop_comparator.config import ConfigError, load_config, parse_config

VALID = {
    "experiment": {"name": "tox21_sr_are", "seed": 42},
    "data": {
        "dataset": "tox21",
        "task": "SR-ARE",
        "fingerprint": {"type": "morgan", "n_bits": 1024, "radius": 2},
        "split": {"train": 0.8, "val": 0.1, "test": 0.1},
    },
    "model": {"hidden_layers": [128, 64], "n_outputs": 2, "snn": {"T": 10, "beta": 0.95, "threshold": 1.0}},
    "training": {"optimizer": "adam", "lr": 1e-4, "batch_size": 16, "epochs": 100},
    "architecture": {
        "weight_bits": 32,
        "e_mac": 1.0,
        "e_ac": 0.2,
        "e_upd": 0.3,
        "memory_scenarios": ["von_neumann", "neuromorphic"],
    },
}


def mutated(path: str, value):
    """A copy of VALID with the dotted `path` set to `value` (None deletes the key)."""
    raw = copy.deepcopy(VALID)
    *parents, last = path.split(".")
    node = raw
    for key in parents:
        node = node[key]
    if value is None:
        del node[last]
    else:
        node[last] = value
    return raw


def test_valid_config_loads_with_agreed_defaults(tmp_path):
    file = tmp_path / "c.yaml"
    file.write_text(yaml.safe_dump(VALID))
    cfg = load_config(file)
    assert cfg.experiment.device == "mps"
    assert cfg.experiment.n_threads == 4
    assert cfg.data.split.method == "stratified"
    assert cfg.model.ann.activation == "relu"
    assert cfg.model.snn.surrogate == "arctan"
    assert cfg.model.snn.reset == "subtract"
    assert cfg.training.patience == 15
    assert cfg.training.class_weights is True
    assert cfg.sweep is None


def test_config_is_immutable():
    cfg = parse_config(VALID)
    with pytest.raises(ValidationError):
        cfg.experiment.seed = 1


@pytest.mark.parametrize(
    "path, value, field",
    [
        ("data.split.train", 0.7, "data.split"),  # proportions do not sum to 1
        ("model.snn.T", 0, "model.snn.T"),
        ("model.snn.beta", 0, "model.snn.beta"),
        ("model.snn.beta", 1.5, "model.snn.beta"),
        ("model.snn.threshold", -1.0, "model.snn.threshold"),
        ("model.hidden_layers", [], "model.hidden_layers"),
        ("model.hidden_layers", [128, 0], "model.hidden_layers.1"),
        ("model.n_outputs", 1, "model.n_outputs"),
        ("data.fingerprint.type", "ecfp", "data.fingerprint.type"),
        ("data.fingerprint.n_bits", 0, "data.fingerprint.n_bits"),
        ("experiment.device", "cuda", "experiment.device"),
        ("experiment.n_threads", -1, "experiment.n_threads"),
        ("training.lr", 0, "training.lr"),
        ("training.optimizer", "sgd", "training.optimizer"),
        ("architecture.memory_scenarios", [], "architecture.memory_scenarios"),
        ("architecture.memory_scenarios", ["gpu"], "architecture.memory_scenarios.0"),
        ("model.snn.treshold", 1.0, "model.snn.treshold"),  # typo in a key
        ("sweep", {"T": [4, -8], "fingerprint": ["maccs"]}, "sweep.T.1"),
    ],
)
def test_invalid_value_names_the_failing_field(path, value, field):
    with pytest.raises(ConfigError) as exc:
        parse_config(mutated(path, value))
    assert field in str(exc.value)


@pytest.mark.parametrize("path", ["experiment.seed", "model.snn", "training.epochs", "architecture"])
def test_missing_field_is_reported(path):
    with pytest.raises(ConfigError) as exc:
        parse_config(mutated(path, None))
    assert path in str(exc.value)


def test_missing_file(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")


def test_invalid_yaml_and_wrong_top_level(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("a: [unclosed")
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_config(bad)
    bad.write_text("- just\n- a list\n")
    with pytest.raises(ConfigError, match="mapping"):
        load_config(bad)


def test_shipped_experiment_config_is_valid():
    from pathlib import Path

    cfg = load_config(Path(__file__).parent.parent / "configs" / "tox21_sr_are.yaml")
    assert cfg.data.task == "SR-ARE"
    assert cfg.model.hidden_layers == [128, 64]
    assert cfg.sweep is not None and cfg.sweep.T == [4, 8, 10, 16]
