from sympy.abc import x,y,z
from dendrite.core.expression import Expression as E
from dendrite.core.functional import Functional as F
from dendrite.mathematics.trigonometry import cos, sin
from dendrite.mathematics.elementary import exp

@E
def schwarz_p(p) -> F:
  return cos(p(x,y,z)*x) + sin(y) + cos(z)

@E
def schwarz_d() -> F:
  return sin(x)*sin(y)*sin(z) + sin(x)*cos(y)*cos(z) + cos(x)*sin(y)*cos(z) + cos(x)*cos(y)*sin(z)

@E
def lidinoid() -> F:
  return 0.5 * (sin(2*x)*cos(y)*sin(z) + sin(2*y)*cos(z)*sin(x) + sin(2*z)*cos(x)*sin(y)) - 0.5 * (cos(2*x)*cos(2*y) + cos(2*y)*cos(2*z) + cos(2*z)*cos(2*x)) + 0.15

@E
def scherks() -> F:
  return (exp(z) * cos(y)) - cos(x)

@E
def neovius() -> F:
  return 3*(cos(x)+cos(y)+cos(z)) + 4*(cos(x)*cos(y)*cos(z))

@E
def gyroid() -> F:
  return sin(x)*cos(y) + sin(y)*cos(z) + sin(z)*cos(x)
