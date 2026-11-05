"""Sensitivity of the SNN advantage to the energy parameters, recomputed from the stored counts (no retraining, D-44)."""

from mac_sop_comparator.comparison.point import PointResult

UPD_RATIOS = [1.0 + 0.25 * i for i in range(11)]  # e_upd / e_ac from 1 to 3.5 (Loihi range, D-21)
BYTE_ENERGIES = [10 ** (-1 + 0.075 * i) for i in range(81)]  # e_byte from 0.1 to 1e5 MAC energies
BYTE_KINDS = ("von_neumann", "neuromorphic")


def cost_ratio_vs_update_energy(point: PointResult, ratios: list[float] = UPD_RATIOS) -> list[float]:
    """Compute-only cost ratio C_SNN / C_ANN when e_upd = ratio * e_ac."""
    m, e_ac = point.metrics, point.params.e_ac
    return [(m.n_sop * e_ac + m.n_upd * k * e_ac) / m.cost_ann for k in ratios]


def energy_ratio_vs_byte_energy(point: PointResult, kind: str, values: list[float] = BYTE_ENERGIES) -> list[float]:
    """Total-energy ratio E_SNN / E_ANN when the energy per byte of one scenario takes each of `values`.

    `von_neumann` varies the ANN's e_byte and `neuromorphic` the SNN's; the other keeps the value of the config.
    """
    if kind not in BYTE_KINDS:
        raise ValueError(f"kind must be one of {BYTE_KINDS}, got '{kind}'")
    m, p = point.metrics, point.params
    if kind == "von_neumann":
        return [m.energy_snn / (m.cost_ann + v * m.bytes_ann) for v in values]
    return [(m.cost_snn + v * m.bytes_snn) / m.energy_ann for v in values]
