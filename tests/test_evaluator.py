import numpy as np
import pytest
import torch
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from mac_sop_comparator.data import MolecularDataset
from mac_sop_comparator.models import AnnClassifier
from mac_sop_comparator.training import Evaluator, collect_outputs, positive_scores


def threshold_model(sign=1.0):
    """A 1-layer model whose logit gap is sign * (x0 - 0.5): predicts class 1 iff x0 = 1 (sign=+1)."""
    model = AnnClassifier([2, 2])
    with torch.no_grad():
        model.layers()[0].weight.copy_(torch.tensor([[-sign / 2, 0.0], [sign / 2, 0.0]]))
        model.layers()[0].bias.copy_(torch.tensor([sign / 4, -sign / 4]))  # no ties: gap = sign * (x0 - 0.5)
    return model


def dataset(n_pos=20, n_neg=80):
    X = np.zeros((n_pos + n_neg, 2), dtype=np.float32)
    X[:n_pos, 0] = 1.0  # positives have x0 = 1
    return MolecularDataset(X, np.array([1] * n_pos + [0] * n_neg))


def test_perfect_model():
    out = Evaluator().evaluate(threshold_model(), dataset())
    assert out == {"roc_auc": 1.0, "balanced_accuracy": 1.0}


def test_inverted_model_is_perfectly_wrong():
    out = Evaluator().evaluate(threshold_model(-1.0), dataset())
    assert out == {"roc_auc": 0.0, "balanced_accuracy": 0.0}


def test_always_negative_model_shows_why_plain_accuracy_misleads():
    model = AnnClassifier([2, 2])
    with torch.no_grad():
        model.layers()[0].weight.zero_()
        model.layers()[0].bias.copy_(torch.tensor([1.0, 0.0]))  # always favors class 0
    data = dataset()  # 80 % negative: plain accuracy would be 0.8
    out = Evaluator().evaluate(model, data)
    assert out["balanced_accuracy"] == 0.5 and out["roc_auc"] == 0.5


def test_matches_scikit_learn_on_random_scores():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    data = MolecularDataset((rng.random((300, 10)) < 0.3).astype(np.float32), rng.integers(0, 2, 300))
    model = AnnClassifier([10, 6, 2])
    outputs, y = collect_outputs(model, data)
    ev = Evaluator()
    assert ev.roc_auc(model, data) == pytest.approx(roc_auc_score(y.numpy(), positive_scores(outputs).numpy()))
    assert ev.balanced_accuracy(model, data) == pytest.approx(balanced_accuracy_score(y.numpy(), outputs.argmax(1).numpy()))


def test_evaluation_does_not_change_the_training_mode_or_the_weights():
    model = threshold_model()
    before = [p.clone() for p in model.parameters()]
    model.train()
    Evaluator().evaluate(model, dataset())
    assert model.training
    model.eval()
    Evaluator().evaluate(model, dataset())
    assert not model.training
    assert all(torch.equal(a, b) for a, b in zip(before, model.parameters()))


def test_handles_more_samples_than_one_eval_batch():
    n = 1300  # > EVAL_BATCH_SIZE
    rng = np.random.default_rng(1)
    data = MolecularDataset((rng.random((n, 2)) < 0.5).astype(np.float32), rng.integers(0, 2, n))
    outputs, y = collect_outputs(threshold_model(), data)
    assert outputs.shape == (n, 2) and len(y) == n


def test_roc_auc_needs_both_classes():
    one_class = MolecularDataset(np.zeros((5, 2), dtype=np.float32), np.zeros(5))
    with pytest.raises(ValueError, match="both classes"):
        Evaluator().roc_auc(threshold_model(), one_class)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_evaluates_a_model_on_mps():
    out = Evaluator().evaluate(threshold_model().to("mps"), dataset())
    assert out["roc_auc"] == 1.0
