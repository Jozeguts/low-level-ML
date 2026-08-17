import math

import pytest

from engine import Value


def numerical_grad(fn, x, eps=1e-6):
    return (fn(x + eps) - fn(x - eps)) / (2 * eps)


def test_addition_backward():
    a = Value(2.0)
    b = Value(3.0)
    y = a + b
    y.backward()
    assert a.grad == pytest.approx(1.0)
    assert b.grad == pytest.approx(1.0)


def test_multiplication_backward():
    a = Value(2.0)
    b = Value(4.0)
    y = a * b
    y.backward()
    assert a.grad == pytest.approx(4.0)
    assert b.grad == pytest.approx(2.0)


def test_shared_node_accumulates_gradient():
    x = Value(3.0)
    y = x * x + x
    y.backward()
    assert x.grad == pytest.approx(7.0)


def test_power_backward():
    x = Value(3.0)
    y = x**3
    y.backward()
    assert x.grad == pytest.approx(27.0)


def test_exp_and_log_backward():
    x = Value(2.0)
    y = x.exp().log()
    y.backward()
    assert x.grad == pytest.approx(1.0)


def test_relu_backward():
    positive = Value(2.0)
    negative = Value(-2.0)
    (positive.relu() + negative.relu()).backward()
    assert positive.grad == pytest.approx(1.0)
    assert negative.grad == pytest.approx(0.0)


def test_finite_difference():
    x = 1.7
    expected = numerical_grad(lambda v: math.exp(v) * v**2, x)
    value = Value(x)
    y = value.exp() * value**2
    y.backward()
    assert value.grad == pytest.approx(expected, rel=1e-5, abs=1e-7)


def test_topological_order_contains_each_node_once():
    x = Value(2.0)
    y = x * x
    z = y + x
    nodes = z.graph_nodes()
    assert len(nodes) == len({id(node) for node in nodes})
    assert nodes[-1] is z
    assert nodes.index(x) < nodes.index(y) < nodes.index(z)


def test_backward_resets_previous_gradients():
    x = Value(2.0)
    y = x * x
    y.backward()
    assert x.grad == pytest.approx(4.0)
    y.backward()
    assert x.grad == pytest.approx(4.0)
