import sympy
from sympy.abc import t
from functools import partial
from numbers import Number
from tensorflow import Tensor
from dendrite.core.operad import Operad
from dendrite.core.algebra import Algebra
from dendrite.core.transformation import Transformation
from dendrite.core.expression import Expression
from dendrite.mathematics.elementary import Max, Min
from dendrite.decorators.type_coercion import convert_to_operad

class Functional(Operad, Algebra):
  def __init__(self, expr, namespace=None):
    if isinstance(expr, Number):
      expression = sympy.sympify(expr)
    else:
      expression = expr
    super().__init__(expression, namespace)

  def __mul__(self, other):
    @Expression
    def mulitply(a, b) -> Functional:
      return a * b
    return mulitply(self, convert_to_functional(other))

  def __truediv__(self, other):
    @Expression
    def division(a, b) -> Functional:
      return a / b
    return division(self, convert_to_functional(other))

  def __pow__(self, other):
    @Expression
    def power(a, b) -> Functional:
      return a ** b
    return power(self, convert_to_functional(other))

  def __abs__(self):
    @Expression
    def absolute(a) -> Functional:
      return abs(a)
    return absolute(self)

  def __invert__(self):
    @Expression
    def complement(a) -> Functional:
      return -a
    return complement(self)

  def __and__(self, other):
    @Expression
    def intersection(a, b) -> Functional:
      return Min(a, b)
    return intersection(self, other)

  def __or__(self, other):
    @Expression
    def union(a, b) -> Functional:
      return Max(a, b)
    return union(self, other)

  def __floordiv__(self, other):
    @Expression
    def subtract(a, b) -> Functional:
      return Min(a, -b)
    return subtract(self, other)

  def __neg__(self):
    return ~self

  def __lshift__(self, other):
    if isinstance(other, Transformation):
      @Expression
      def composition(f, g: sympy.Tuple) -> Functional:
        # Hack due to Subs not doing simultaneous substitution
        return sympy.Subs(sympy.Subs(f, ('x', 'y', 'z'), ('fx', 'fy', 'fz')), ('fx', 'fy', 'fz'), g)
      return composition(self, other)
    else:
      super().__lshift__(other)

convert_to_functional = partial(convert_to_operad, Functional)
