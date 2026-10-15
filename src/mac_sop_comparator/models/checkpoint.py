"""Save and load trained models (.pt). Only tensors and plain values are stored (safe `weights_only` load)."""

from pathlib import Path

import torch

from mac_sop_comparator.models.ann import AnnClassifier
from mac_sop_comparator.models.base import BaseClassifier
from mac_sop_comparator.models.snn import SnnClassifier


def save_checkpoint(model: BaseClassifier, path: str | Path) -> Path:
    if isinstance(model, AnnClassifier):
        data = {"kind": "ann", "topology": list(model.topology), "activation": model.activation_name}
    elif isinstance(model, SnnClassifier):
        data = {
            "kind": "snn",
            "topology": list(model.topology),
            "T": model.T,
            "beta": model.beta,
            "threshold": model.threshold,
            "surrogate": model.surrogate_name,
            "reset": model.reset,
        }
    else:
        raise TypeError(f"cannot save a {type(model).__name__}: only AnnClassifier and SnnClassifier are supported")
    data["state_dict"] = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, path)
    return path


def load_checkpoint(path: str | Path) -> BaseClassifier:
    """Rebuild the model from the file alone; it is returned on the CPU."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    data = torch.load(path, map_location="cpu", weights_only=True)
    kind = data.get("kind")
    if kind == "ann":
        model = AnnClassifier(data["topology"], data["activation"])
    elif kind == "snn":
        model = SnnClassifier(
            data["topology"], data["T"], data["beta"], data["threshold"], data["surrogate"], data["reset"]
        )
    else:
        raise ValueError(f"{path}: unsupported model kind {kind!r}")
    model.load_state_dict(data["state_dict"])
    return model
