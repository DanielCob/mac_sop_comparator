"""Configuration schema (immutable pydantic models) with its validation rules."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

FingerprintType = Literal["maccs", "morgan"]
PositiveInt = Annotated[int, Field(gt=0)]


class _Section(BaseModel):
    """Base for every section: immutable, and unknown keys are errors (catches typos)."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class ExperimentConfig(_Section):
    name: str = Field(min_length=1)
    seed: int = Field(ge=0)
    device: Literal["mps", "cpu"] = "mps"
    n_threads: int = Field(default=4, ge=0)  # CPU only; 0 = all cores


class FingerprintConfig(_Section):
    type: FingerprintType
    n_bits: int = Field(gt=0)  # ignored for maccs (166 bits, bit 0 dropped)
    radius: int = Field(ge=0)  # ignored for maccs


class SplitConfig(_Section):
    train: float = Field(gt=0, lt=1)
    val: float = Field(gt=0, lt=1)
    test: float = Field(gt=0, lt=1)
    method: Literal["stratified"] = "stratified"

    @model_validator(mode="after")
    def _proportions_sum_to_one(self) -> "SplitConfig":
        total = self.train + self.val + self.test
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"train + val + test must sum to 1, got {total:g}")
        return self


class DataConfig(_Section):
    dataset: str = Field(min_length=1)
    task: str = Field(min_length=1)
    fingerprint: FingerprintConfig
    split: SplitConfig


class AnnConfig(_Section):
    activation: Literal["relu", "tanh", "sigmoid"] = "relu"


class SnnConfig(_Section):
    T: int = Field(gt=0)
    beta: float = Field(gt=0, le=1)
    threshold: float = Field(gt=0)
    surrogate: Literal["arctan"] = "arctan"
    reset: Literal["subtract", "zero"] = "subtract"


class ModelConfig(_Section):
    # The input size is derived from the fingerprint (D-05), so it is not declared here.
    hidden_layers: list[PositiveInt] = Field(min_length=1)
    n_outputs: int = Field(ge=2)
    snn: SnnConfig
    ann: AnnConfig = Field(default_factory=AnnConfig)


class TrainingConfig(_Section):
    optimizer: Literal["adam"]
    lr: float = Field(gt=0)
    batch_size: int = Field(gt=0)
    epochs: int = Field(gt=0)
    patience: int = Field(default=15, gt=0)
    class_weights: bool = True


class ArchitectureParams(_Section):
    weight_bits: int = Field(gt=0)
    e_mac: float = Field(gt=0)
    e_ac: float = Field(ge=0)
    e_upd: float = Field(ge=0)
    memory_scenarios: list[Literal["von_neumann", "neuromorphic"]] = Field(min_length=1)


class SweepConfig(_Section):
    T: list[PositiveInt] = Field(min_length=1)
    fingerprint: list[FingerprintType] = Field(min_length=1)
    n_bits: list[PositiveInt] | None = Field(default=None, min_length=1)  # morgan only; None = not swept
    max_points: int = Field(default=50, gt=0)


class Config(_Section):
    experiment: ExperimentConfig
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    architecture: ArchitectureParams
    sweep: SweepConfig | None = None
