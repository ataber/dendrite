from sympy.abc import x,y,z
from dendrite.core.transformation import Transformation as T
from dendrite.core.expression import Expression
from dendrite.mathematics.elementary import sqrt
from dendrite.mathematics.trigonometry import arctan, arccos

@Expression
def cylindrical() -> T:
  fz = sqrt(x**2 + y**2)
  fy = z
  fx = arctan(y/x)
  return (fx,fy,fz)

@Expression
def spherical() -> T:
  r = sqrt(x**2 + y**2 + z**2)
  fz = r
  fy = arccos(z/r)
  fx = arctan(y/x)
  return (fx,fy,fz)
