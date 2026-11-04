"""EfficiencyMetrics: unified metrics of the ANN/SNN comparison from two OpReports (Table 12, D-30)."""

import math
from dataclasses import asdict, dataclass

from mac_sop_comparator.config import ArchitectureParams
from mac_sop_comparator.profiling import OpReport, TrafficModel


@dataclass(frozen=True)
class MetricSet:
    """Operation counts, activity rates and relative energy costs of one ANN/SNN pair.

    `r` is the effective rate SOPs / (T * N_MAC), the one compared with `critical_rate`; `rate_per_layer`
    is spikes / (neurons * T) for each layer, input included (D-30). Costs are in the units of the
    energies of `ArchitectureParams`; they count operations only. `energy_*` add the memory term
    e_byte * bytes (D-37), with each network in its own scenario (D-24); `ai_*` are operations per byte.
    """

    n_mac: float
    n_sop: float
    n_upd: float
    T: int
    r: float
    rate_per_layer: list[float]
    cost_ann: float
    cost_snn: float
    cost_ratio: float  # C_SNN / C_ANN, below 1 means the SNN is cheaper
    ops_ratio: float  # (N_SOP + N_UPD) / N_MAC
    r_critical: float
    snn_cheaper: bool
    bytes_ann: float
    bytes_snn: float
    ai_ann: float  # N_MAC / bytes_ann
    ai_snn: float  # (N_SOP + N_UPD) / bytes_snn
    energy_ann: float  # cost_ann + e_byte_von_neumann * bytes_ann
    energy_snn: float  # cost_snn + e_byte_neuromorphic * bytes_snn
    energy_ratio: float  # energy_snn / energy_ann

    def to_dict(self) -> dict:
        return asdict(self)


def critical_rate(params: ArchitectureParams, n_mac: float, n_upd: float, T: int) -> float:
    """r* = (e_mac*N_MAC - e_upd*N_UPD) / (T*e_ac*N_MAC): the SNN is cheaper while r < r*.

    It is negative when the state updates alone cost more than the whole ANN, and infinite when SOPs are free.
    """
    numerator = params.e_mac * n_mac - params.e_upd * n_upd
    denominator = T * params.e_ac * n_mac
    if denominator == 0:
        return math.copysign(math.inf, numerator) if numerator else 0.0
    return numerator / denominator


class EfficiencyMetrics:
    """Combines an ANN report and an SNN report with the architecture parameters."""

    @staticmethod
    def compute(
        ann: OpReport, snn: OpReport, params: ArchitectureParams, T: int, layer_sizes: list[int]
    ) -> MetricSet:
        """`layer_sizes` = [N_0 (input), N_1, ..., N_L] of the topology both reports were measured on."""
        if ann.architecture != "ann" or snn.architecture != "snn":
            raise ValueError(f"expected an ANN and an SNN report, got '{ann.architecture}' and '{snn.architecture}'")
        if T <= 0:
            raise ValueError(f"T must be > 0, got {T}")
        if len(snn.spikes_per_layer) != len(layer_sizes):
            raise ValueError(f"{len(snn.spikes_per_layer)} spike counts for {len(layer_sizes)} layer sizes")
        n_mac = ann.macs
        if n_mac == 0:
            raise ValueError("the ANN has 0 MACs")
        n_sop, n_upd = snn.sops, snn.state_updates
        cost_ann = n_mac * params.e_mac
        cost_snn = n_sop * params.e_ac + n_upd * params.e_upd
        traffic = TrafficModel(params, layer_sizes)
        bytes_ann, bytes_snn = traffic.bytes_accessed(ann), traffic.bytes_accessed(snn)
        energy_ann = cost_ann + traffic.e_byte("ann") * bytes_ann
        energy_snn = cost_snn + traffic.e_byte("snn") * bytes_snn
        return MetricSet(
            n_mac=n_mac,
            n_sop=n_sop,
            n_upd=n_upd,
            T=T,
            r=n_sop / (T * n_mac),
            rate_per_layer=[s / (n * T) for s, n in zip(snn.spikes_per_layer, layer_sizes)],
            cost_ann=cost_ann,
            cost_snn=cost_snn,
            cost_ratio=cost_snn / cost_ann,
            ops_ratio=(n_sop + n_upd) / n_mac,
            r_critical=critical_rate(params, n_mac, n_upd, T),
            snn_cheaper=cost_snn < cost_ann,
            bytes_ann=bytes_ann,
            bytes_snn=bytes_snn,
            ai_ann=traffic.intensity(n_mac, bytes_ann),
            ai_snn=traffic.intensity(n_sop + n_upd, bytes_snn),
            energy_ann=energy_ann,
            energy_snn=energy_snn,
            energy_ratio=energy_snn / energy_ann,
        )
