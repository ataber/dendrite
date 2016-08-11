from sympy.abc import x,y,z
from Core.Transformation import Transformation as T
from Core.Expression import Expression
from Mathematics.Elementary import sqrt
from Mathematics.Trigonometry import arctan, arccos

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
