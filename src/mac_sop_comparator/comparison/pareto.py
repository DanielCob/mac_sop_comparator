"""ParetoAnalyzer: non-dominated models in (test ROC-AUC, cost) over all the points of a sweep (D-44)."""

import csv
from dataclasses import dataclass
from pathlib import Path

from mac_sop_comparator.comparison.point import PointResult

METRICS = ("cost", "energy")  # compute only, or compute plus memory traffic


@dataclass(frozen=True)
class Candidate:
    """One trained model of a point: higher `auc` and lower `cost` / `energy` are better."""

    candidate_id: str  # `<point_id>:ann` or `<point_id>:snn`
    point_id: str
    architecture: str
    group: str  # points that differ only in T share it; their ANN is the same model
    auc: float
    cost: float
    energy: float

    def value(self, metric: str) -> float:
        return getattr(self, metric)


class ParetoAnalyzer:
    @staticmethod
    def candidates(results: list[PointResult]) -> list[Candidate]:
        """The ANN and the SNN of every point; an ANN shared by several T is listed once (the first)."""
        if not results:
            raise ValueError("there are no results to analyze")
        candidates, seen_anns = [], set()
        for p in results:
            if p.group not in seen_anns:
                seen_anns.add(p.group)
                candidates.append(
                    Candidate(f"{p.point_id}:ann", p.point_id, "ann", p.group, p.ann_auc, p.metrics.cost_ann, p.metrics.energy_ann)
                )
            candidates.append(
                Candidate(f"{p.point_id}:snn", p.point_id, "snn", p.group, p.snn_auc, p.metrics.cost_snn, p.metrics.energy_snn)
            )
        return candidates

    @staticmethod
    def frontier(candidates: list[Candidate], metric: str = "cost") -> list[Candidate]:
        """Candidates no other one beats in both AUC and `metric` (at least one strictly), cheapest first.

        Equal candidates do not dominate each other, so both stay on the frontier.
        """
        if metric not in METRICS:
            raise ValueError(f"metric must be one of {METRICS}, got '{metric}'")

        def dominates(b: Candidate, a: Candidate) -> bool:
            return b.auc >= a.auc and b.value(metric) <= a.value(metric) and (b.auc > a.auc or b.value(metric) < a.value(metric))

        front = [a for a in candidates if not any(dominates(b, a) for b in candidates)]
        return sorted(front, key=lambda c: (c.value(metric), -c.auc))

    @classmethod
    def write_csv(cls, results: list[PointResult], path: str | Path) -> Path:
        """Every candidate with its values and whether it is on each frontier."""
        candidates = cls.candidates(results)
        on = {metric: {c.candidate_id for c in cls.frontier(candidates, metric)} for metric in METRICS}
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["candidate_id", "point_id", "architecture", "auc", "cost", "energy", "on_frontier_cost", "on_frontier_energy"])
            for c in candidates:
                writer.writerow([c.candidate_id, c.point_id, c.architecture, c.auc, c.cost, c.energy,
                                 c.candidate_id in on["cost"], c.candidate_id in on["energy"]])  # fmt: skip
        return path
