import pytest
import torch

from mac_sop_comparator.models import AnnClassifier, SnnClassifier, load_checkpoint, save_checkpoint


def test_roundtrip_gives_identical_outputs(tmp_path):
    torch.manual_seed(0)
    model = AnnClassifier([32, 16, 8, 2], "tanh").eval()
    path = save_checkpoint(model, tmp_path / "sub" / "ann.pt")
    loaded = load_checkpoint(path)
    assert loaded.topology == [32, 16, 8, 2] and loaded.activation_name == "tanh"
    x = torch.rand(10, 32)
    assert torch.equal(model(x), loaded(x))


def test_a_model_on_mps_is_saved_and_reloaded_on_the_cpu(tmp_path):
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available")
    model = AnnClassifier([8, 4, 2]).to("mps")
    loaded = load_checkpoint(save_checkpoint(model, tmp_path / "ann.pt"))
    assert next(loaded.parameters()).device.type == "cpu"
    assert torch.allclose(model(torch.ones(1, 8, device="mps")).cpu(), loaded(torch.ones(1, 8)), atol=1e-6)


def test_checkpoint_holds_only_plain_data(tmp_path):
    path = save_checkpoint(AnnClassifier([8, 4, 2]), tmp_path / "ann.pt")
    data = torch.load(path, weights_only=True)  # fails if it contained arbitrary pickled objects
    assert set(data) == {"kind", "topology", "activation", "state_dict"}


def test_errors(tmp_path):
    with pytest.raises(FileNotFoundError, match="checkpoint not found"):
        load_checkpoint(tmp_path / "nope.pt")
    torch.save({"kind": "cnn"}, tmp_path / "cnn.pt")
    with pytest.raises(ValueError, match="unsupported model kind"):
        load_checkpoint(tmp_path / "cnn.pt")
    with pytest.raises(TypeError, match="only AnnClassifier and SnnClassifier"):
        save_checkpoint(torch.nn.Linear(2, 2), tmp_path / "x.pt")


def test_snn_roundtrip_keeps_every_hyperparameter_and_gives_identical_counts(tmp_path):
    torch.manual_seed(0)
    model = SnnClassifier([32, 16, 2], T=6, beta=0.9, threshold=1.5, surrogate_gradient="arctan", reset="zero")
    loaded = load_checkpoint(save_checkpoint(model, tmp_path / "snn.pt"))
    assert isinstance(loaded, SnnClassifier)
    assert (loaded.topology, loaded.T, loaded.beta, loaded.threshold, loaded.reset, loaded.surrogate_name) == (
        [32, 16, 2], 6, 0.9, 1.5, "zero", "arctan",
    )
    x = (torch.rand(8, 32) < 0.5).float()
    assert torch.equal(model.eval()(x), loaded.eval()(x))


def test_snn_checkpoint_holds_only_plain_data(tmp_path):
    path = save_checkpoint(SnnClassifier([8, 4, 2], 3, 0.95, 1.0), tmp_path / "snn.pt")
    data = torch.load(path, weights_only=True)
    assert set(data) == {"kind", "topology", "T", "beta", "threshold", "surrogate", "reset", "state_dict"}
