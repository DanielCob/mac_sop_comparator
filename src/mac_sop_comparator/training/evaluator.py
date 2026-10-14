"""Classification metrics (ROC-AUC and balanced accuracy) for any model on any partition."""

import torch
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from mac_sop_comparator.data import MolecularDataset, batches
from mac_sop_comparator.models import BaseClassifier

EVAL_BATCH_SIZE = 512


@torch.no_grad()
def collect_outputs(model: BaseClassifier, data: MolecularDataset) -> tuple[torch.Tensor, torch.Tensor]:
    """Run `model` over `data` in eval mode (on the model's device); return (outputs, labels) on the CPU."""
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    outputs, labels = [], []
    try:
        for x, y in batches(data, EVAL_BATCH_SIZE):
            outputs.append(model(x.to(device)).cpu())
            labels.append(y)
    finally:
        model.train(was_training)
    return torch.cat(outputs), torch.cat(labels)


def positive_scores(outputs: torch.Tensor) -> torch.Tensor:
    """Score of the positive class from the model outputs (softmax over logits)."""
    return torch.softmax(outputs, dim=1)[:, 1]


class Evaluator:
    def roc_auc(self, model: BaseClassifier, data: MolecularDataset) -> float:
        outputs, y = collect_outputs(model, data)
        return self._roc_auc(y, outputs)

    def balanced_accuracy(self, model: BaseClassifier, data: MolecularDataset) -> float:
        outputs, y = collect_outputs(model, data)
        return self._balanced_accuracy(y, outputs)

    def evaluate(self, model: BaseClassifier, data: MolecularDataset) -> dict[str, float]:
        """Both metrics from a single pass over the data."""
        outputs, y = collect_outputs(model, data)
        return {"roc_auc": self._roc_auc(y, outputs), "balanced_accuracy": self._balanced_accuracy(y, outputs)}

    @staticmethod
    def _roc_auc(y: torch.Tensor, outputs: torch.Tensor) -> float:
        if len(set(y.tolist())) < 2:
            raise ValueError("the ROC-AUC needs both classes in the evaluated data")
        return float(roc_auc_score(y.numpy(), positive_scores(outputs).numpy()))

    @staticmethod
    def _balanced_accuracy(y: torch.Tensor, outputs: torch.Tensor) -> float:
        return float(balanced_accuracy_score(y.numpy(), outputs.argmax(dim=1).numpy()))
