from sympy.abc import x,y,z
from Core.Transformation import Transformation as T
from Core.Expression import Expression
from Mathematics.Elementary import sqrt, square
from Mathematics.Trigonometry import arctan, arccos

@Expression
def cylindrical() -> T:
  fz = sqrt(square(y) + square(x))
  fy = z
  fx = arctan(y/x)
  return (fx,fy,fz)

@Expression
def spherical() -> T:
  r = sqrt(square(y) + square(x) + square(z))
  fz = r
  fy = arccos(z/r)
  fx = arctan(y/x)
  return (fx,fy,fz)
