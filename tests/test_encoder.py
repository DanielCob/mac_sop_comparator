import pytest
import torch

from mac_sop_comparator.config import SnnConfig
from mac_sop_comparator.data import SpikeEncoder


def test_replicates_a_batch_exactly():
    x = torch.tensor([[1.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    spikes = SpikeEncoder(5).rate_encode(x)
    assert spikes.shape == (5, 2, 3) and spikes.dtype == torch.float32
    for t in range(5):
        assert torch.equal(spikes[t], x)


def test_single_sample_has_shape_T_F():
    spikes = SpikeEncoder(4).rate_encode(torch.tensor([1.0, 0.0]))
    assert spikes.shape == (4, 2) and spikes.sum().item() == 4


def test_is_deterministic_and_independent_of_the_global_seed():
    x = torch.randint(0, 2, (8, 16)).float()
    enc = SpikeEncoder(6)
    torch.manual_seed(0)
    a = enc.rate_encode(x)
    torch.manual_seed(999)
    assert torch.equal(a, enc.rate_encode(x))


def test_total_spikes_equal_T_times_active_bits():
    x = torch.randint(0, 2, (8, 16)).float()
    assert SpikeEncoder(7).rate_encode(x).sum().item() == 7 * x.sum().item()


def test_integer_input_is_accepted_and_output_is_independent_memory():
    spikes = SpikeEncoder(3).rate_encode(torch.tensor([[1, 0]]))
    assert spikes.dtype == torch.float32
    spikes[0] += 5  # must not alias the other time steps
    assert torch.equal(spikes[1], torch.tensor([[1.0, 0.0]]))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_stays_on_the_input_device():
    x = torch.tensor([[1.0, 0.0]], device="mps")
    assert SpikeEncoder(2).rate_encode(x).device.type == "mps"


def test_rejects_non_binary_and_bad_shapes():
    enc = SpikeEncoder(2)
    with pytest.raises(ValueError, match="binary"):
        enc.rate_encode(torch.tensor([[0.5, 1.0]]))
    with pytest.raises(ValueError, match="shape"):
        enc.rate_encode(torch.zeros(2, 3, 4))
    with pytest.raises(ValueError, match="T must be"):
        SpikeEncoder(0)


def test_from_config():
    assert SpikeEncoder.from_config(SnnConfig(T=10, beta=0.95, threshold=1.0)).T == 10
