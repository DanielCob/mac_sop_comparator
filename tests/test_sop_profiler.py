import numpy as np
import pytest
import torch

from mac_sop_comparator.data import MolecularDataset
from mac_sop_comparator.models import AnnClassifier, SnnClassifier
from mac_sop_comparator.profiling import MacProfiler, SopProfiler


def drive(model, weights):
    """Set each Linear layer to a constant weight (list per layer) and zero bias."""
    with torch.no_grad():
        for fc, w in zip(model.fcs, weights):
            fc.weight.fill_(w)
            fc.bias.zero_()


def data_with(bits_per_sample, n_features):
    X = np.zeros((len(bits_per_sample), n_features), dtype=np.float32)
    for i, k in enumerate(bits_per_sample):
        X[i, :k] = 1.0
    return MolecularDataset(X, np.zeros(len(X)))


def test_hand_computed_example():
    # [4, 3, 2], T = 5: layer 1 fires on every step (3 neurons), layer 2 is silent (weights 0)
    model = SnnClassifier([4, 3, 2], T=5, beta=0.95, threshold=1.0)
    drive(model, [1.0, 0.0])
    report = SopProfiler(model).measure(data_with([2, 2], 4))  # k = 2 active bits in both samples
    per_step = 2 * 3 + 3 * 2  # S_0*N_1 + S_1*N_2 = 12
    assert report.architecture == "snn"
    assert report.sops == pytest.approx(5 * per_step)  # 60
    assert report.state_updates == 5 * (3 + 2)  # 25
    assert report.spikes_per_layer == pytest.approx([5 * 2, 5 * 3, 0])  # input, layer 1, output (L + 1 values)
    assert report.biases == 3 + 2 and report.macs is None


def test_the_output_layer_spikes_have_no_fanout_and_are_not_counted_as_sops():
    model = SnnClassifier([4, 3, 2], T=4, beta=0.95, threshold=1.0)
    drive(model, [1.0, 1.0])  # everything fires: the output emits 4 * 2 spikes per sample
    report = SopProfiler(model).measure(data_with([4], 4))
    assert report.spikes_per_layer == pytest.approx([16, 12, 8])
    assert report.sops == pytest.approx(16 * 3 + 12 * 2)  # S_0*N_1 + S_1*N_2: the 8 output spikes add nothing


def test_the_average_is_per_sample():
    model = SnnClassifier([4, 3, 2], T=5, beta=0.95, threshold=1.0)
    drive(model, [1.0, 0.0])
    report = SopProfiler(model).measure(data_with([2, 4], 4))  # 60 and 5 * (4*3 + 6) = 90 SOPs
    assert report.sops == pytest.approx(75.0)
    assert report.spikes_per_layer[0] == pytest.approx(5 * 3)  # mean of 10 and 20 input spikes


def test_no_activity_means_no_sops_but_the_states_still_update():
    model = SnnClassifier([8, 6, 2], T=5, beta=0.95, threshold=1.0)
    drive(model, [0.0, 0.0])
    report = SopProfiler(model).measure(data_with([3, 5], 8))
    assert report.spikes_per_layer == pytest.approx([5 * 4, 0, 0])  # the input spikes still exist...
    assert report.sops == pytest.approx(5 * 4 * 6)  # ...and each one reaches the 6 neurons of layer 1
    assert report.state_updates == 5 * (6 + 2)


def test_full_activity_reaches_T_times_the_number_of_connections():
    model = SnnClassifier([1024, 128, 64, 2], T=10, beta=0.95, threshold=1.0)
    drive(model, [1.0, 1.0, 1.0])
    report = SopProfiler(model).measure(data_with([1024], 1024))
    macs = MacProfiler(AnnClassifier([1024, 128, 64, 2])).measure().macs
    assert report.sops == pytest.approx(10 * macs) and macs == 139_392
    assert report.state_updates == 10 * (128 + 64 + 2)


def test_it_does_not_depend_on_the_batch_size_and_restores_the_model_mode(monkeypatch):
    import mac_sop_comparator.profiling.profiler as module

    torch.manual_seed(0)
    model = SnnClassifier([16, 8, 2], T=6, beta=0.95, threshold=1.0)
    data = MolecularDataset((np.random.default_rng(0).random((50, 16)) < 0.4).astype(np.float32), np.zeros(50))
    model.train()
    full = SopProfiler(model).measure(data)
    assert model.training
    monkeypatch.setattr(module, "PROFILE_BATCH_SIZE", 7)
    assert SopProfiler(model).measure(data) == full


def test_errors():
    snn = SnnClassifier([4, 2], 3, 0.95, 1.0)
    with pytest.raises(ValueError, match="needs data"):
        SopProfiler(snn).measure()
    with pytest.raises(TypeError, match="SnnClassifier"):
        SopProfiler(AnnClassifier([4, 2]))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_profiles_a_model_on_mps():
    model = SnnClassifier([4, 3, 2], T=5, beta=0.95, threshold=1.0).to("mps")
    drive(model, [1.0, 0.0])
    assert SopProfiler(model).measure(data_with([2, 2], 4)).sops == pytest.approx(60.0)
