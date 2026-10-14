"""OpReport: operation counts per inference, with the JSON schema of the design (D-04 names, D-16 rule)."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

ARCHITECTURES = ("ann", "snn")
MEMORY_SCENARIOS = ("von_neumann", "neuromorphic")


@dataclass(frozen=True)
class OpReport:
    """Operations of one inference (SNN values are averages per sample).

    ANN: `macs`. SNN: `sops`, `state_updates` and `spikes_per_layer`. `biases` is the number of bias terms,
    reported apart because they count neither as MACs nor as SOPs (D-16). `bytes_accessed` maps a memory
    scenario to the bytes moved; it stays None until the traffic model exists (phase 5).
    """

    architecture: str
    macs: int | None = None
    sops: float | None = None
    state_updates: float | None = None
    spikes_per_layer: list[float] | None = None
    biases: int | None = None
    bytes_accessed: dict[str, float] | None = None

    def __post_init__(self) -> None:
        if self.architecture not in ARCHITECTURES:
            raise ValueError(f"architecture must be one of {ARCHITECTURES}, got '{self.architecture}'")
        ann_fields = {"macs": self.macs}
        snn_fields = {"sops": self.sops, "state_updates": self.state_updates, "spikes_per_layer": self.spikes_per_layer}
        own, other = (ann_fields, snn_fields) if self.architecture == "ann" else (snn_fields, ann_fields)
        missing = [k for k, v in own.items() if v is None]
        if missing:
            raise ValueError(f"a {self.architecture.upper()} report needs {missing}")
        extra = [k for k, v in other.items() if v is not None]
        if extra:
            raise ValueError(f"a {self.architecture.upper()} report cannot have {extra}")
        for name in ("macs", "sops", "state_updates", "biases"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must be >= 0, got {value}")
        if self.spikes_per_layer is not None and any(s < 0 for s in self.spikes_per_layer):
            raise ValueError("spikes_per_layer must be >= 0")
        if self.bytes_accessed is not None:
            unknown = set(self.bytes_accessed) - set(MEMORY_SCENARIOS)
            if unknown:
                raise ValueError(f"bytes_accessed scenarios must be in {MEMORY_SCENARIOS}, got {sorted(unknown)}")

    def to_dict(self) -> dict:
        """The JSON schema: {"architecture": ..., "per_inference": {...}}."""
        values = asdict(self)
        return {"architecture": values.pop("architecture"), "per_inference": values}

    @classmethod
    def from_dict(cls, data: dict) -> "OpReport":
        return cls(architecture=data["architecture"], **data["per_inference"])

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "OpReport":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
