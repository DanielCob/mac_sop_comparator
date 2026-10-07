import dataclasses

import pytest

from mac_sop_comparator.config import (
    ArchitectureParams,
    Config,
    DataConfig,
    ExperimentConfig,
    FingerprintConfig,
    ModelConfig,
    SnnConfig,
    SplitConfig,
    TrainingConfig,
)


@pytest.fixture
def config() -> Config:
    return Config(
        experiment=ExperimentConfig(name="tox21_sr_are", seed=42),
        data=DataConfig(
            dataset="tox21",
            task="SR-ARE",
            fingerprint=FingerprintConfig(type="morgan", n_bits=1024, radius=2),
            split=SplitConfig(train=0.8, val=0.1, test=0.1),
        ),
        model=ModelConfig(
            hidden_layers=[128, 64],
            n_outputs=2,
            snn=SnnConfig(T=10, beta=0.95, threshold=1.0),
        ),
        training=TrainingConfig(optimizer="adam", lr=1e-4, batch_size=16, epochs=100),
        architecture=ArchitectureParams(
            weight_bits=32,
            e_mac=1.0,
            e_ac=0.2,
            e_upd=0.3,
            memory_scenarios=["von_neumann", "neuromorphic"],
        ),
    )


def test_agreed_defaults(config):
    assert config.experiment.device == "mps"
    assert config.experiment.n_threads == 4
    assert config.data.split.method == "stratified"
    assert config.model.ann.activation == "relu"
    assert config.model.snn.surrogate == "arctan"
    assert config.model.snn.reset == "subtract"
    assert config.training.patience == 15
    assert config.training.class_weights is True
    assert config.sweep is None


def test_config_is_immutable(config):
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.experiment.seed = 1
