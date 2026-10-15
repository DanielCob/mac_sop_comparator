"""Mini-batch training with a weighted cross-entropy, validation-based model selection and early stopping."""

import copy
import logging
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from mac_sop_comparator.config import TrainingConfig
from mac_sop_comparator.data import MolecularDataset, batches
from mac_sop_comparator.models import BaseClassifier
from mac_sop_comparator.training.evaluator import Evaluator, collect_outputs

logger = logging.getLogger(__name__)

OPTIMIZERS = {"adam": torch.optim.Adam}


def compute_class_weights(y: np.ndarray, n_classes: int = 2) -> torch.Tensor:
    """Inverse-frequency weights n / (n_classes * n_c), so every class weighs the same in total (D-14)."""
    counts = np.bincount(y, minlength=n_classes)
    if (counts == 0).any():
        raise ValueError(f"every class needs at least one training sample, got counts {counts.tolist()}")
    return torch.tensor(len(y) / (n_classes * counts), dtype=torch.float32)


@dataclass
class TrainingResult:
    history: list[dict] = field(default_factory=list)  # per epoch: epoch, train_loss, val_loss, val_auc
    best_epoch: int = 0
    best_val_auc: float = float("nan")
    stopped_early: bool = False


class Trainer:
    def __init__(
        self,
        optimizer: str = "adam",
        lr: float = 1e-4,
        batch_size: int = 16,
        epochs: int = 100,
        patience: int = 15,
        class_weights: bool = True,
        device: str | torch.device = "cpu",
        seed: int = 42,
    ):
        if optimizer not in OPTIMIZERS:
            raise ValueError(f"optimizer must be one of {sorted(OPTIMIZERS)}, got '{optimizer}'")
        self.optimizer_name, self.lr, self.batch_size = optimizer, lr, batch_size
        self.epochs, self.patience, self.class_weights = epochs, patience, class_weights
        self.device, self.seed = torch.device(device), seed

    @classmethod
    def from_config(cls, config: TrainingConfig, device: str | torch.device, seed: int) -> "Trainer":
        return cls(
            config.optimizer, config.lr, config.batch_size, config.epochs, config.patience,
            config.class_weights, device, seed,
        )

    def train(self, model: BaseClassifier, train: MolecularDataset, val: MolecularDataset) -> TrainingResult:
        """Train `model` in place; on return it holds the weights of the best validation ROC-AUC epoch."""
        model.to(self.device)
        weight = compute_class_weights(train.y, model.n_outputs).to(self.device) if self.class_weights else None
        criterion = nn.CrossEntropyLoss(weight=weight)
        optimizer = OPTIMIZERS[self.optimizer_name](model.parameters(), lr=self.lr)
        generator = torch.Generator().manual_seed(self.seed)  # one generator: a new order every epoch, reproducible

        result, best_state, stale = TrainingResult(), None, 0
        for epoch in range(1, self.epochs + 1):
            train_loss = self._fit_epoch(model, train, criterion, optimizer, generator)
            val_loss, val_auc = self._validate(model, val, criterion)
            result.history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_auc": val_auc})
            logger.debug("epoch %d: train %.4f, val %.4f, val AUC %.4f", epoch, train_loss, val_loss, val_auc)
            if best_state is None or val_auc > result.best_val_auc:
                result.best_epoch, result.best_val_auc, stale = epoch, val_auc, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                stale += 1
                if stale >= self.patience:
                    result.stopped_early = True
                    break
        model.load_state_dict(best_state)
        logger.info(
            "trained %d epochs%s; best epoch %d with validation ROC-AUC %.4f",
            len(result.history), " (early stop)" if result.stopped_early else "", result.best_epoch, result.best_val_auc,
        )
        return result

    def _fit_epoch(self, model, data, criterion, optimizer, generator) -> float:
        model.train()
        total, n = torch.zeros((), device=self.device), 0  # accumulated on the device: no sync per batch
        for x, y in batches(data, self.batch_size, shuffle=True, generator=generator):
            x, y = x.to(self.device), y.to(self.device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total, n = total + loss.detach() * len(y), n + len(y)
        return total.item() / n

    def _validate(self, model, data, criterion) -> tuple[float, float]:
        logits, scores, y = collect_outputs(model, data)
        loss = criterion(logits.to(self.device), y.to(self.device)).item()
        return loss, Evaluator._roc_auc(y, scores)
