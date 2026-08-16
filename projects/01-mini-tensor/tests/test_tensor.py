import numpy as np

from mini_tensor.tensor import Tensor


def test_shape_and_strides():
    x = Tensor(np.arange(6).reshape(2, 3))
    assert x.shape == (2, 3)
    assert x.strides == (3, 1)


def test_broadcast_add():
    x = Tensor([[1, 2], [3, 4]], requires_grad=True)
    b = Tensor([10, 20], requires_grad=True)
    y = (x + b).sum()
    y.backward()
    np.testing.assert_allclose(y.data, 38)
    np.testing.assert_allclose(x.grad, [[1, 1], [1, 1]])
    np.testing.assert_allclose(b.grad, [2, 2])


def test_mul_gradient():
    x = Tensor([2.0, 3.0], requires_grad=True)
    y = (x * x).sum()
    y.backward()
    np.testing.assert_allclose(x.grad, [4.0, 6.0])


def test_matmul_gradient():
    x = Tensor([[1.0, 2.0]], requires_grad=True)
    w = Tensor([[3.0], [4.0]], requires_grad=True)
    y = (x @ w).sum()
    y.backward()
    np.testing.assert_allclose(x.grad, [[3.0, 4.0]])
    np.testing.assert_allclose(w.grad, [[1.0], [2.0]])


def test_relu():
    x = Tensor([-1.0, 0.0, 2.0], requires_grad=True)
    y = x.relu().sum()
    y.backward()
    np.testing.assert_allclose(y.data, 2.0)
    np.testing.assert_allclose(x.grad, [0.0, 0.0, 1.0])


def test_reshape_gradient():
    x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = x.reshape(3, 1).sum()
    y.backward()
    np.testing.assert_allclose(x.grad, [1.0, 1.0, 1.0])
