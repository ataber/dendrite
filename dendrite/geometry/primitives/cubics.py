from sympy.abc import x,y,z
from dendrite.core.functional import Functional as F
from dendrite.core.expression import Expression as E

# from "Real-Time Ray Tracing of Implicit Surfaceson the GPU", Jag Mohan Singh and P.J. Narayanan
@E
def ding_dong() -> F:
  return x**2 + y**2 - z*(1-z**2)

@E
def cayley() -> F:
  return -5*((x**2)*(y+z) + (y**2)*(x+z) + (z**2)*(x+y)) + 2*(x*y + y*x + x*z)
