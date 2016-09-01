from sympy.abc import x,y,z
from dendrite.core.transformation import Transformation as T
from dendrite.core.functional import Functional as F
from dendrite.core.expression import Expression as E
from dendrite.transformations.domain import scale_inputs
from dendrite.mathematics.trigonometry import sin, cos

@E
def twist(theta) -> T:
  c = cos(theta)
  s = sin(theta)
  return (x*c - y*s, x*s + y*c, z)

@E
def taper(r) -> T:
  return scale_inputs(r, r, 1)()
