import sympy
from sympy.abc import x,y,z
from dendrite.core.operad import Operad
from dendrite.core.expression import Expression

class Transformation(Operad):
  def __init__(self, expression, namespace=None):
    super().__init__(expression, namespace)

  def __lshift__(self, other):
    if isinstance(other, Transformation):
      @Expression
      def composition(f, g: sympy.Tuple) -> Transformation:
        # Hack due to Subs not doing simultaneous substitution
        return sympy.Subs(sympy.Subs(f, ('x', 'y', 'z'), ('fx', 'fy', 'fz')), ('fx', 'fy', 'fz'), g)
      return composition(self, other)
    else:
      super().__lshift__(other)
