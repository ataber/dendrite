from dendrite.core.vector_space import VectorSpace

class Algebra(VectorSpace):
  def __mul__(self, other):
    raise NotImplementedError

  def __truediv__(self, other):
    raise NotImplementedError
