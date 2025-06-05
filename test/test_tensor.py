from tensor import Tensor
import pytest

def test_add():
  assert Tensor(2.0) + Tensor(3.0) == Tensor(5.0)
