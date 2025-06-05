from tensor import Tensor
import pytest

def test_add():
  assert Tensor(2.0) + Tensor(3.0) == Tensor(5.0)
  assert Tensor([2.0, 3.0]) + Tensor([3.0, 4.0]) == Tensor([5.0, 7.0])
  assert Tensor([[1.0, 2.0], [3.0, 4.0]]) + Tensor([[5.0, 6.0], [7.0, 8.0]]) == Tensor([[6.0, 8.0], [10.0, 12.0]])
