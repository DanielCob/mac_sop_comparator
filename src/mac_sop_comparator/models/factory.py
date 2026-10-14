"""Builds the ANN and its homologous SNN from the same configuration (Factory Method, D-22)."""

from mac_sop_comparator.config import ModelConfig
from mac_sop_comparator.models.ann import AnnClassifier
from mac_sop_comparator.models.base import BaseClassifier

KINDS = ("ann", "snn")


class ModelFactory:
    @staticmethod
    def topology(model_cfg: ModelConfig, n_inputs: int) -> list[int]:
        """[n_inputs, *hidden_layers, n_outputs]; the input size comes from the fingerprint (D-05)."""
        if n_inputs < 1:
            raise ValueError(f"n_inputs must be >= 1, got {n_inputs}")
        return [n_inputs, *model_cfg.hidden_layers, model_cfg.n_outputs]

    @staticmethod
    def build(model_cfg: ModelConfig, n_inputs: int, kind: str) -> BaseClassifier:
        if kind not in KINDS:
            raise ValueError(f"model kind must be one of {KINDS}, got '{kind}'")
        topology = ModelFactory.topology(model_cfg, n_inputs)
        if kind == "ann":
            return AnnClassifier(topology, model_cfg.ann.activation)
        raise NotImplementedError("the SNN classifier is implemented in phase 4 (P4.1)")
