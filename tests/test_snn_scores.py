import pytest
import torch

from mac_sop_comparator.data import MolecularDataset
from mac_sop_comparator.models import AnnClassifier, SnnClassifier
from mac_sop_comparator.training import Evaluator

import numpy as np


def scores_for(counts, gaps, T=10):
    """positive_scores of an SNN whose last forward produced `counts` (n, 2) with output membrane gaps m1 - m0."""
    model = SnnClassifier([2, 2], T, 0.95, 1.0)
    counts = torch.tensor(counts, dtype=torch.float32)
    model.membrane_sum = torch.stack([torch.zeros(len(gaps)), torch.tensor(gaps, dtype=torch.float32)], dim=1)
    return model.positive_scores(counts)


def test_ann_scores_are_the_softmax_in_float64():
    outputs = torch.tensor([[0.0, 0.0], [1.0, 3.0], [3.0, 1.0]])
    s = AnnClassifier([2, 2]).positive_scores(outputs)
    assert s.dtype == torch.float64
    assert s.tolist() == pytest.approx(torch.softmax(outputs, dim=1)[:, 1].tolist())


def test_equal_counts_and_no_membrane_gap_give_one_half():
    assert scores_for([[0, 0], [3, 3]], [0.0, 0.0]).tolist() == pytest.approx([0.5, 0.5])


def test_without_tie_break_it_is_softmax_of_the_counts():
    s = scores_for([[2, 5]], [0.0])
    assert s.item() == pytest.approx(torch.softmax(torch.tensor([2.0, 5.0]), dim=0)[1].item())


def test_membrane_orders_samples_with_the_same_counts():
    s = scores_for([[0, 0]] * 4, [-5.0, -0.1, 0.1, 5.0])
    assert s.tolist() == sorted(s.tolist()) and len(set(s.tolist())) == 4
    assert s[0] < 0.5 < s[3]


def test_membrane_never_reorders_samples_with_different_count_differences():
    # count difference d = c1 - c0: any sample with a larger d must score higher, whatever its membrane
    low = scores_for([[0, 4]], [-1e6]).item()  # d = 4, worst possible membrane
    high = scores_for([[0, 5]], [+1e6]).item()  # d = 5 ... and a better one
    worse_membrane_high = scores_for([[0, 5]], [-1e6]).item()  # d = 5, worst membrane
    best_membrane_low = scores_for([[0, 4]], [+1e6]).item()  # d = 4, best membrane
    assert worse_membrane_high > best_membrane_low and high > low


def test_huge_membrane_gaps_are_still_distinguished():
    s = scores_for([[0, 0]] * 2, [1000.0, 2000.0])
    assert s[1] > s[0]


def test_scores_are_probabilities_in_float64():
    s = scores_for([[0, 10], [10, 0], [3, 3]], [4.0, -4.0, 0.0])
    assert s.dtype == torch.float64 and ((s > 0) & (s < 1)).all()


def test_must_follow_the_forward_pass():
    model = SnnClassifier([2, 2], 4, 0.95, 1.0)
    with pytest.raises(RuntimeError, match="right after"):
        model.positive_scores(torch.zeros(3, 2))
    model(torch.ones(3, 2))
    with pytest.raises(RuntimeError, match="right after"):
        model.positive_scores(torch.zeros(5, 2))  # not the outputs of that forward pass


def test_the_evaluator_uses_the_membrane_to_break_ties_between_silent_outputs():
    """Output neurons never reach the threshold (counts all 0), but the membrane still carries the signal."""
    model = SnnClassifier([2, 2], T=3, beta=0.95, threshold=100.0)
    with torch.no_grad():
        model.fcs[0].weight.copy_(torch.tensor([[0.0, 0.0], [1.0, 0.0]]))  # output 1 integrates x0
        model.fcs[0].bias.zero_()
    X = np.zeros((20, 2), dtype=np.float32)
    X[:5, 0] = 1.0
    data = MolecularDataset(X, np.array([1] * 5 + [0] * 15))
    out = Evaluator().evaluate(model, data)
    assert out["roc_auc"] == 1.0  # ordered by membrane even though every count is 0
    assert out["balanced_accuracy"] == 0.5  # D-29: the class is the argmax of the counts, and they all tie at 0


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_scores_work_on_mps():
    model = SnnClassifier([2, 2], 4, 0.95, 1.0).to("mps")
    out = model(torch.ones(3, 2, device="mps"))
    assert model.positive_scores(out).dtype == torch.float64
