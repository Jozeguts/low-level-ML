import pytest
import torch


# The extension build/loading will be added as the next increment.
# These tests define the contract for the native operator.


def reference(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.add(a, b)


def test_reference_contract():
    a = torch.tensor([1.0, -2.0, 3.5], dtype=torch.float32)
    b = torch.tensor([4.0, 5.0, -1.5], dtype=torch.float32)
    expected = reference(a, b)
    assert expected.dtype == torch.float32
    assert torch.equal(expected, torch.tensor([5.0, 3.0, 2.0]))


def test_shapes_must_match():
    a = torch.ones(4, dtype=torch.float32)
    b = torch.ones(2, dtype=torch.float32)
    with pytest.raises(RuntimeError):
        reference(a, b)


def test_empty_tensor():
    a = torch.empty(0, dtype=torch.float32)
    b = torch.empty(0, dtype=torch.float32)
    out = reference(a, b)
    assert out.numel() == 0


def test_multidimensional_input():
    a = torch.randn(8, 16, dtype=torch.float32)
    b = torch.randn(8, 16, dtype=torch.float32)
    torch.testing.assert_close(reference(a, b), a + b)
