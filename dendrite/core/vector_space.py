from dendrite.core.expression import Expression

class VectorSpace:
  def __add__(self, other):
    @Expression
    def addition(a, b) -> self.__class__:
      return a + b
    return addition(self, other)

  def __sub__(self, other):
    @Expression
    def subtraction(a, b) -> self.__class__:
      return a - b
    return subtraction(self, other)
