import logging

import pytest
import torch

from mac_sop_comparator.utils import get_device, set_num_threads, set_seed, setup_logging


def test_package_imports():
    import mac_sop_comparator

    assert mac_sop_comparator.__version__


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_same_seed_gives_same_tensors(device):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS not available")
    set_seed(42)
    a = torch.rand(5, device=device)
    set_seed(42)
    b = torch.rand(5, device=device)
    assert torch.equal(a, b)


def test_different_seeds_differ():
    set_seed(1)
    a = torch.rand(5)
    set_seed(2)
    assert not torch.equal(a, torch.rand(5))


def test_set_num_threads():
    set_num_threads(2)
    assert torch.get_num_threads() == 2
    with pytest.raises(ValueError, match="n_threads"):
        set_num_threads(0)


def test_get_device_cpu_and_invalid():
    assert get_device("cpu").type == "cpu"
    with pytest.raises(ValueError, match="experiment.device"):
        get_device("cuda")


def test_get_device_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert get_device("mps").type == "cpu"


def test_setup_logging_writes_file_with_versions(tmp_path):
    log_file = tmp_path / "sub" / "run.log"
    setup_logging(log_file)
    logging.getLogger("test").info("hello")
    for h in logging.getLogger().handlers:
        h.flush()
    text = log_file.read_text()
    assert "torch " in text and "snntorch " in text and "hello" in text
