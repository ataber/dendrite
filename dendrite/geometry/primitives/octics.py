from sympy.abc import x,y,z
from dendrite.core.functional import Functional as F
from dendrite.core.expression import Expression as E

# from "Real-Time Ray Tracing of Implicit Surfaceson the GPU", Jag Mohan Singh and P.J. Narayanan
def chmutov():
  T_8 = lambda t: 128*(t**8) - 256*(t**6) + 160*(t**4) - 32*(t**2) + 1
  return (T_8(x) + T_8(y) + T_8(z)) * -1
