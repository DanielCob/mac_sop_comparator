import numpy as np
import pytest
import torch

from mac_sop_comparator.data import MolecularDataset, batches


@pytest.fixture
def ds():
    X = np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 1, 1, 0], [0, 0, 0, 0], [1, 1, 1, 1]] * 2)
    return MolecularDataset(X, np.arange(10) % 2)


def test_basic_properties(ds):
    assert len(ds) == 10 and ds.n_features == 4
    x, y = ds[2]
    assert x.tolist() == [0, 1, 1, 0] and y == 0
    assert ds.X.dtype == np.float32 and ds.y.dtype == np.int64


def test_sparsity_is_fraction_of_zeros(ds):
    assert ds.sparsity() == pytest.approx(1 - 14 / 40)  # 7 ones per block of 5 rows, 2 blocks, 40 cells
    assert MolecularDataset(np.zeros((3, 5)), [0, 1, 0]).sparsity() == 1.0
    assert MolecularDataset(np.ones((3, 5)), [0, 1, 0]).sparsity() == 0.0
    assert MolecularDataset(np.zeros((0, 5)), []).sparsity() == 0.0


def test_subset(ds):
    sub = ds.subset(np.array([4, 0]))
    assert len(sub) == 2 and sub[0][0].tolist() == [1, 1, 1, 1] and sub.sparsity() == pytest.approx(1 - 5 / 8)


def test_shape_validation():
    with pytest.raises(ValueError, match="2-D"):
        MolecularDataset(np.zeros(5), np.zeros(5))
    with pytest.raises(ValueError, match="shape"):
        MolecularDataset(np.zeros((5, 2)), np.zeros(4))


def test_batches_sizes_and_types(ds):
    out = list(batches(ds, 4))
    assert [len(x) for x, _ in out] == [4, 4, 2]  # last batch is smaller
    x, y = out[0]
    assert x.dtype == torch.float32 and y.dtype == torch.int64 and x.shape == (4, 4)


def test_no_shuffle_keeps_order(ds):
    ys = torch.cat([y for _, y in batches(ds, 3)])
    assert ys.tolist() == ds.y.tolist()


def test_shuffle_covers_every_sample_once(ds):
    g = torch.Generator().manual_seed(0)
    seen = torch.cat([x[:, 0] for x, _ in batches(ds, 3, shuffle=True, generator=g)])
    assert sorted(seen.tolist()) == sorted(ds.X[:, 0].tolist())


def test_shuffle_is_reproducible_and_changes_each_epoch():
    ds = MolecularDataset(np.arange(40, dtype=np.float32).reshape(20, 2), np.zeros(20))

    def epochs(seed):
        g = torch.Generator().manual_seed(seed)
        return [torch.cat([x[:, 0] for x, _ in batches(ds, 4, shuffle=True, generator=g)]).tolist() for _ in range(2)]

    a, b = epochs(42), epochs(42)
    assert a == b  # same seed, same sequence of epochs
    assert a[0] != a[1]  # the order differs between epochs
    assert a[0] != epochs(7)[0]


def test_batch_arguments_are_validated(ds):
    with pytest.raises(ValueError, match="batch_size"):
        list(batches(ds, 0))
    with pytest.raises(ValueError, match="Generator"):
        list(batches(ds, 2, shuffle=True))
