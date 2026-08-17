import numpy as np
import pytest

from mini_tensor.storage import Storage, TensorLayout
from mini_tensor.tensor import Tensor


def test_contiguous_layout():
    layout = TensorLayout((2, 3), (3, 1))
    assert layout.numel == 6
    assert layout.is_contiguous()
    assert layout.offset((1, 2)) == 5


def test_transpose_changes_strides_without_copy():
    x = Tensor(np.arange(6).reshape(2, 3))
    y = x.T
    assert y.shape == (3, 2)
    assert y.strides == (1, 3)
    assert not y.is_contiguous
    assert y.storage is x.storage
    assert y.data_ptr == x.data_ptr
    np.testing.assert_array_equal(y.data, [[0, 3], [1, 4], [2, 5]])


def test_slice_is_view_and_tracks_offset():
    x = Tensor(np.arange(12).reshape(3, 4))
    y = x[1:, 1:3]
    assert y.shape == (2, 2)
    assert y.strides == (4, 1)
    assert y.storage is x.storage
    assert y.storage_offset == 5
    np.testing.assert_array_equal(y.data, [[5, 6], [9, 10]])


def test_view_backward_scatter():
    x = Tensor(np.arange(6, dtype=float).reshape(2, 3), requires_grad=True)
    y = x[:, 1:]
    y.sum().backward()
    np.testing.assert_array_equal(x.grad, [[0, 1, 1], [0, 1, 1]])


def test_transpose_backward_restores_layout():
    x = Tensor(np.arange(6, dtype=float).reshape(2, 3), requires_grad=True)
    x.T.sum().backward()
    np.testing.assert_array_equal(x.grad, np.ones((2, 3)))


def test_storage_requires_flat_buffer():
    with pytest.raises(ValueError):
        Storage(np.zeros((2, 2)))
