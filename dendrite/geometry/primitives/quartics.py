from sympy.abc import x,y,z
from dendrite.core.functional import Functional as F
from dendrite.core.expression import Expression as E

# from "Real-Time Ray Tracing of Implicit Surfaceson the GPU", Jag Mohan Singh and P.J. Narayanan
@E
def cushion() -> F:
  return ((z**2)*(x**2) - z**4 - 2*z*(x**2) +
         2*(z**3) + x**2 - z**2 - (x**2 - z)**2 -
         y**4 - 2*(x**2)*(y**2) - (y**2)*(z**2) + 2*(y**2)*z + (y**2))

@E
def goursat() -> F:
  return 1 - x**4 - y**4 - z**4

@E
def tooth() -> F:
  return x**2 + y**2 + z**2 - x**4 - y**4 - z**4
