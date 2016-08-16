from numpy import ndarray
from sympy.abc import x,y,z
from dendrite.core.functional import Functional as F
from dendrite.core.expression import Expression as E

@E
def plane(a, b, c, offset: float) -> F:
  return x*a + b*y + c*z + offset
