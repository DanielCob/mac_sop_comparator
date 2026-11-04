"""TrafficModel: bytes moved per inference and arithmetic intensity, one memory scenario per network (D-24, D-42)."""

from dataclasses import replace

from mac_sop_comparator.config import ArchitectureParams
from mac_sop_comparator.profiling.report import MEMORY_SCENARIOS, OpReport

# Scenario each network is evaluated in (D-24): the ANN reads weights from main memory, the SNN keeps them on chip.
SCENARIO_OF = {"ann": "von_neumann", "snn": "neuromorphic"}


class TrafficModel:
    """Bytes of one inference from an OpReport, the layer sizes and the memory parameters.

    ANN, von Neumann: B = b_w*P/B_lote + b_a*A, with P weights plus biases and A the neurons of every layer.
    SNN, neuromorphic: B = b_e*(S_0 + ... + S_(L-1)) + b_o*N_L: the events that feed a layer plus the output values;
    weights and neuron states stay local. SNN, von Neumann (not used by the reports): B = b_w*N_SOP + 2*b_s*N_UPD.
    """

    def __init__(self, params: ArchitectureParams, layer_sizes: list[int]) -> None:
        if len(layer_sizes) < 2 or any(n <= 0 for n in layer_sizes):
            raise ValueError(f"layer_sizes needs an input and an output layer with neurons, got {layer_sizes}")
        self.params = params
        self.layer_sizes = list(layer_sizes)

    def bytes_accessed(self, report: OpReport, scenario: str | None = None) -> float:
        """Bytes moved by one inference of `report`'s network; `scenario` defaults to the one of D-24."""
        scenario = scenario or SCENARIO_OF[report.architecture]
        if scenario not in MEMORY_SCENARIOS:
            raise ValueError(f"scenario must be one of {MEMORY_SCENARIOS}, got '{scenario}'")
        p = self.params
        if report.architecture == "ann":
            if scenario != "von_neumann":
                raise ValueError("the ANN is only evaluated in the von Neumann scenario (D-24)")
            weights = report.macs + (report.biases or 0)
            return p.b_w * weights / p.inference_batch + p.b_a * sum(self.layer_sizes)
        if scenario == "neuromorphic":
            if len(report.spikes_per_layer) != len(self.layer_sizes):
                raise ValueError(f"{len(report.spikes_per_layer)} spike counts for {len(self.layer_sizes)} layer sizes")
            return p.b_e * sum(report.spikes_per_layer[:-1]) + p.b_o * self.layer_sizes[-1]
        return p.b_w * report.sops + 2 * p.b_s * report.state_updates

    def e_byte(self, architecture: str) -> float:
        """Energy per byte of the scenario assigned to the network."""
        if SCENARIO_OF[architecture] == "von_neumann":
            return self.params.e_byte_von_neumann
        return self.params.e_byte_neuromorphic

    @staticmethod
    def intensity(n_ops: float, n_bytes: float) -> float:
        """Arithmetic intensity: operations per byte moved."""
        if n_bytes <= 0:
            raise ValueError(f"bytes must be > 0, got {n_bytes}")
        return n_ops / n_bytes

    def annotate(self, report: OpReport) -> OpReport:
        """A copy of `report` with `bytes_accessed` filled: the assigned scenario, None in the other (design schema)."""
        assigned = SCENARIO_OF[report.architecture]
        values = {s: (self.bytes_accessed(report, s) if s == assigned else None) for s in MEMORY_SCENARIOS}
        return replace(report, bytes_accessed=values)
