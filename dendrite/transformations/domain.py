from sympy.abc import x,y,z
from dendrite.core.expression import Expression as E
from dendrite.core.transformation import Transformation as T

@E
def scale_inputs(sx, sy, sz) -> T:
  return (x / sx, y / sy, z / sz)

@E
def absolute_value() -> T:
  return (abs(x), abs(y), abs(z))
