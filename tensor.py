import numpy as np

class Tensor:
  def __init__(self, data: np.ndarray, _children=(),op='', label=''):
    self.data = np.asarray(data, dtype=float)
    self.grad = np.zeros_like(self.data)
    self._backward = lambda: None
    self._prev = set(_children)
    self._op = op
    self.label = label

  def __repr__(self):
    return f'Tensor(data={self.data})'

  def __eq__(self, other):
    return bool(np.array_equal(self.data, other.data))

  def __hash__(self):
    return hash((self.data.shape, self.data.tobytes()))

  def __add__(self, other):
    other = other if isinstance(other, Tensor) else Tensor(other)
    out = Tensor(self.data + other.data, (self, other), '+')
    def _backward():
      self.grad += out.grad
      other.grad += out.grad
    out._backward = _backward
    return out

  def __radd__(self, other):
    return self + other
    
  def __sub__(self, other):
    return self + (-other)

  def __rsub__(self, other):
    return other + (-self)

  def __neg__(self):
    out = Tensor(-self.data, (self,), 'neg')
    def _backward():
      self.grad += -out.grad
    out._backward = _backward
    return out  

  def __mul__(self, other):
    other = other if isinstance(other, Tensor) else Tensor(other)
    out = Tensor(self.data * other.data, (self, other), '*')
    def _backward():
      self.grad += other.data * out.grad
      other.grad += self.data * out.grad
    out._backward = _backward
    return out

  def __rmul__(self, other):
    return self * other

  def __truediv__(self, other):
    other = other if isinstance(other, Tensor) else Tensor(other)
    return self * other.pow(-1)

  def __rtruediv__(self, other):
    return other * self.pow(-1)

  def pow(self, exponent): # TODO how does this work
    """
    Element‐wise power; exponent can be scalar or array.
    """
    out = Tensor(self.data**exponent, (self,), f'**{exponent}')

    def _backward():
        self.grad += (exponent * (self.data ** (exponent - 1))) * out.grad
    out._backward = _backward
    return out

  def __pow__(self, exponent):
    return self.pow(exponent)

  def __matmul__(self, other): # TODO how does this work
    assert isinstance(other, Tensor)
    out = Tensor(self.data @ other.data, (self, other), '*')
    def _backward():
      self.grad += other.data * out.grad
      other.grad += self.data * out.grad

  
  def backward(self):
    topo = []
    visited = set()
    def build_topo(v):
      if v not in visited:
        visited.add(v)
        for child in v._prev:
          build_topo(child)
        topo.append(v)
    build_topo(self)
    self.grad = np.ones_like(self.data)
    for node in reversed(topo):
      node._backward()
