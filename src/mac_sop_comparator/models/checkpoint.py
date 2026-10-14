"""Save and load trained models (.pt). Only tensors and plain values are stored (safe `weights_only` load)."""

from pathlib import Path

import torch

from mac_sop_comparator.models.ann import AnnClassifier
from mac_sop_comparator.models.base import BaseClassifier


def save_checkpoint(model: BaseClassifier, path: str | Path) -> Path:
    if not isinstance(model, AnnClassifier):
        raise TypeError(f"cannot save a {type(model).__name__}: only the ANN is supported so far")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "kind": "ann",
            "topology": list(model.topology),
            "activation": model.activation_name,
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        },
        path,
    )
    return path


def load_checkpoint(path: str | Path) -> BaseClassifier:
    """Rebuild the model from the file alone; it is returned on the CPU."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    data = torch.load(path, map_location="cpu", weights_only=True)
    if data.get("kind") != "ann":
        raise ValueError(f"{path}: unsupported model kind {data.get('kind')!r}")
    model = AnnClassifier(data["topology"], data["activation"])
    model.load_state_dict(data["state_dict"])
    return model
