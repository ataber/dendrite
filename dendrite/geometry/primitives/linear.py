from sympy.abc import x,y,z
from dendrite.core.functional import Functional as F
from dendrite.core.expression import Expression as E
from dendrite.mathematics.metric import norm
from dendrite.decorators.type_coercion import three_vector

@E
def plane(normal: three_vector, offset) -> F:
  normal = normal / norm(normal)
  a, b, c = normal
  return x*a + b*y + c*z + offset
