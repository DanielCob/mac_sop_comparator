"""Configuration schema (dataclasses). Loading and validation live in `config.loader`."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    seed: int
    device: str = "mps"
    n_threads: int = 4


@dataclass(frozen=True)
class FingerprintConfig:
    type: str  # "maccs" | "morgan"
    n_bits: int  # ignored for maccs (166 bits, bit 0 dropped)
    radius: int  # ignored for maccs


@dataclass(frozen=True)
class SplitConfig:
    train: float
    val: float
    test: float
    method: str = "stratified"


@dataclass(frozen=True)
class DataConfig:
    dataset: str
    task: str
    fingerprint: FingerprintConfig
    split: SplitConfig


@dataclass(frozen=True)
class AnnConfig:
    activation: str = "relu"


@dataclass(frozen=True)
class SnnConfig:
    T: int
    beta: float
    threshold: float
    surrogate: str = "arctan"
    reset: str = "subtract"


@dataclass(frozen=True)
class ModelConfig:
    hidden_layers: list[int]  # the input size is derived from the fingerprint
    n_outputs: int
    snn: SnnConfig
    ann: AnnConfig = field(default_factory=AnnConfig)


@dataclass(frozen=True)
class TrainingConfig:
    optimizer: str
    lr: float
    batch_size: int
    epochs: int
    patience: int = 15
    class_weights: bool = True


@dataclass(frozen=True)
class ArchitectureParams:
    weight_bits: int
    e_mac: float
    e_ac: float
    e_upd: float
    memory_scenarios: list[str]  # "von_neumann" | "neuromorphic"


@dataclass(frozen=True)
class SweepConfig:
    T: list[int]
    fingerprint: list[str]
    max_points: int = 50


@dataclass(frozen=True)
class Config:
    experiment: ExperimentConfig
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    architecture: ArchitectureParams
    sweep: SweepConfig | None = None
