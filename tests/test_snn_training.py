import numpy as np
import pytest
import torch
import torch.nn.functional as F
from snntorch import functional as SF

from mac_sop_comparator.data import MolecularDataset
from mac_sop_comparator.models import SnnClassifier
from mac_sop_comparator.training import Evaluator, Trainer, compute_class_weights


def make_data(n=400, n_features=16, seed=0):
    """Binary fingerprints; the label is 1 when at least 2 of the first 3 bits are set (learnable)."""
    rng = np.random.default_rng(seed)
    X = (rng.random((n, n_features)) < 0.3).astype(np.float32)
    return MolecularDataset(X, (X[:, :3].sum(1) >= 2).astype(np.int64))


def fresh_snn(seed=0, T=8):
    torch.manual_seed(seed)
    return SnnClassifier([16, 16, 2], T, beta=0.95, threshold=1.0)


def test_count_cross_entropy_equals_snntorch_ce_count_loss_without_weights():
    """D-35: with no class weights, F.cross_entropy on the counts is exactly `ce_count_loss`."""
    torch.manual_seed(0)
    T, B = 10, 32
    spike_record = (torch.rand(T, B, 2) < 0.4).float()  # (T, B, classes) as the SNN emits them
    y = torch.randint(0, 2, (B,))
    counts = spike_record.sum(dim=0)
    assert F.cross_entropy(counts, y).item() == pytest.approx(SF.ce_count_loss()(spike_record, y).item())


def test_class_weights_change_the_count_loss_as_intended():
    counts = torch.tensor([[3.0, 1.0], [1.0, 3.0], [2.0, 0.0]])
    y = torch.tensor([0, 1, 1])  # the third sample is wrong and belongs to the rare class
    w = compute_class_weights(np.array([0, 0, 0, 0, 1]))  # class 1 is rare => heavier
    assert F.cross_entropy(counts, y, weight=w).item() > F.cross_entropy(counts, y).item()


def test_the_trainer_learns_a_separable_problem_with_the_snn():
    model = fresh_snn()
    res = Trainer(lr=1e-2, epochs=40, patience=40, batch_size=16).train(model, make_data(400, seed=1), make_data(200, seed=2))
    assert res.best_val_auc > 0.9
    assert res.history[-1]["train_loss"] < res.history[0]["train_loss"]
    assert Evaluator().evaluate(model, make_data(200, seed=3))["roc_auc"] > 0.9  # the best epoch was restored


def test_the_snn_training_is_reproducible():
    def run(seed):
        return Trainer(lr=1e-2, epochs=4, seed=seed).train(fresh_snn(), make_data(seed=1), make_data(150, seed=2)).history

    assert run(42) == run(42)
    assert run(42) != run(7)


def test_gradients_flow_through_time_to_the_first_layer():
    model = fresh_snn()
    x, y = torch.from_numpy(make_data(32).X), torch.from_numpy(make_data(32).y)
    F.cross_entropy(model(x), y).backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_trains_on_mps():
    res = Trainer(lr=1e-2, epochs=25, patience=25, device="mps").train(fresh_snn(), make_data(300, seed=1), make_data(100, seed=2))
    assert res.best_val_auc > 0.85
