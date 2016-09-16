import sympy
from sympy.abc import x,y,z
from dendrite.core.operad import Operad
from dendrite.core.expression import Expression
from dendrite.decorators.type_coercion import functional_lambda, transformation_lambda

class Transformation(Operad):
  def __init__(self, expression, namespace=None):
    super().__init__(expression, namespace)

  def __lshift__(self, other):
    if isinstance(other, Transformation):
      @Expression
      def composition(f: transformation_lambda, g: transformation_lambda) -> Functional:
        return sympy.Subs(f.expr, f.variables, g.expr)
      return composition(self, other)
    else:
      super().__lshift__(other)
