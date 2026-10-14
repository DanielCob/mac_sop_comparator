import numpy as np
import pytest
import torch

from mac_sop_comparator.config import TrainingConfig
from mac_sop_comparator.data import MolecularDataset
from mac_sop_comparator.models import AnnClassifier
from mac_sop_comparator.training import Trainer, compute_class_weights


def make_data(n=400, n_features=16, noise=0.0, seed=0, positive_rate=0.3):
    """Binary fingerprints where the label is determined by the first 3 bits (plus optional label noise)."""
    rng = np.random.default_rng(seed)
    X = (rng.random((n, n_features)) < 0.3).astype(np.float32)
    y = (X[:, :3].sum(1) >= 2).astype(np.int64)
    flip = rng.random(n) < noise
    return MolecularDataset(X, np.where(flip, 1 - y, y))


def fresh_model(seed=0):
    torch.manual_seed(seed)
    return AnnClassifier([16, 16, 2])


def test_class_weights_are_inverse_frequency():
    w = compute_class_weights(np.array([0] * 80 + [1] * 20))
    assert w.tolist() == pytest.approx([100 / (2 * 80), 100 / (2 * 20)])
    assert w[1] > w[0]
    with pytest.raises(ValueError, match="every class"):
        compute_class_weights(np.zeros(5, dtype=int))


def test_learns_a_separable_problem():
    model = fresh_model()
    res = Trainer(lr=1e-2, epochs=60, patience=60, batch_size=16).train(model, make_data(400, seed=1), make_data(200, seed=2))
    assert res.best_val_auc > 0.95
    assert res.history[-1]["train_loss"] < res.history[0]["train_loss"]
    assert [h["epoch"] for h in res.history] == list(range(1, len(res.history) + 1))
    assert set(res.history[0]) == {"epoch", "train_loss", "val_loss", "val_auc"}


def test_model_is_restored_to_the_best_epoch():
    model = fresh_model()
    val = make_data(200, noise=0.2, seed=2)
    res = Trainer(lr=1e-2, epochs=40, patience=40).train(model, make_data(400, noise=0.2, seed=1), val)
    trainer = Trainer(lr=1e-2)
    _, auc = trainer._validate(model, val, torch.nn.CrossEntropyLoss())
    assert auc == pytest.approx(res.best_val_auc)
    assert res.best_val_auc == max(h["val_auc"] for h in res.history)
    assert res.history[res.best_epoch - 1]["val_auc"] == res.best_val_auc


def test_early_stopping_waits_for_the_patience():
    # Unlearnable labels: validation AUC stalls, so training must stop long before 200 epochs.
    rng = np.random.default_rng(0)
    noise = lambda n: MolecularDataset((rng.random((n, 16)) < 0.3).astype(np.float32), rng.integers(0, 2, n))
    res = Trainer(lr=1e-3, epochs=200, patience=3).train(fresh_model(), noise(200), noise(100))
    assert res.stopped_early and len(res.history) < 200
    assert len(res.history) == res.best_epoch + 3  # exactly `patience` epochs without improvement


def test_runs_all_epochs_when_it_keeps_improving():
    res = Trainer(lr=1e-2, epochs=3, patience=15).train(fresh_model(), make_data(300, seed=1), make_data(100, seed=2))
    assert len(res.history) == 3 and not res.stopped_early


def test_same_seed_same_history_different_seed_differs():
    def run(seed):
        return Trainer(lr=1e-2, epochs=5, seed=seed).train(fresh_model(), make_data(seed=1), make_data(150, seed=2)).history

    assert run(42) == run(42)
    assert run(42) != run(7)


def test_class_weights_flag_changes_training():
    train, val = make_data(400, noise=0.1, seed=1), make_data(150, noise=0.1, seed=2)
    a = Trainer(lr=1e-2, epochs=3, class_weights=True).train(fresh_model(), train, val).history
    b = Trainer(lr=1e-2, epochs=3, class_weights=False).train(fresh_model(), train, val).history
    assert a[-1]["train_loss"] != b[-1]["train_loss"]


def test_from_config_and_validation():
    t = Trainer.from_config(TrainingConfig(optimizer="adam", lr=1e-4, batch_size=16, epochs=100), "cpu", seed=5)
    assert (t.lr, t.batch_size, t.epochs, t.patience, t.class_weights, t.seed) == (1e-4, 16, 100, 15, True, 5)
    with pytest.raises(ValueError, match="optimizer"):
        Trainer(optimizer="sgd")


def test_validation_needs_both_classes():
    val = MolecularDataset(np.zeros((10, 16), dtype=np.float32), np.zeros(10))
    with pytest.raises(ValueError, match="both classes"):
        Trainer(epochs=1).train(fresh_model(), make_data(), val)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_trains_on_mps():
    model = fresh_model()
    res = Trainer(lr=1e-2, epochs=20, patience=20, device="mps").train(model, make_data(300, seed=1), make_data(100, seed=2))
    assert res.best_val_auc > 0.9 and next(model.parameters()).device.type == "mps"


def test_training_curve_figure(tmp_path):
    from mac_sop_comparator.training import plot_training_curve

    res = Trainer(lr=1e-2, epochs=4).train(fresh_model(), make_data(200, seed=1), make_data(100, seed=2))
    path = plot_training_curve(res.history, res.best_epoch, tmp_path / "sub" / "curve.png", "demo")
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


REAL = __import__("pathlib").Path(__file__).parent.parent / "data_raw" / "tox21.csv.gz"


@pytest.mark.skipif(not REAL.is_file(), reason="tox21.csv.gz not cached in data_raw/")
def test_smoke_training_converges_on_a_small_real_subset():
    """P3.9: with the project's own hyperparameters the ANN fits 300 real molecules (loss falls a lot)."""
    from pathlib import Path

    from mac_sop_comparator.config import load_config
    from mac_sop_comparator.data import prepare_data
    from mac_sop_comparator.models import ModelFactory

    cfg = load_config(Path(__file__).parent.parent / "configs" / "tox21_sr_are.yaml")
    prepared = prepare_data(cfg)
    train = prepared.partition("train").subset(np.arange(300))
    val = prepared.partition("val")
    torch.manual_seed(cfg.experiment.seed)
    model = ModelFactory.build(cfg.model, prepared.X.shape[1], "ann")
    t = cfg.training
    res = Trainer(t.optimizer, t.lr, t.batch_size, 30, 30, t.class_weights, "cpu", cfg.experiment.seed).train(model, train, val)
    first, last = res.history[0]["train_loss"], res.history[-1]["train_loss"]
    assert last < 0.25 * first  # it memorizes the small subset
    assert res.best_val_auc > 0.6  # and it learned something that transfers
