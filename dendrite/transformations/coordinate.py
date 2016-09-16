from sympy import diff, sympify, solve, integrate
from sympy.solvers.solveset import invert_real
from sympy.abc import x,y,z,t
import numpy as np
from dendrite.core.transformation import Transformation as T
from dendrite.core.time_dependent_transformation import TimeDependentTransformation as TDT
from dendrite.core.functional import Functional as F
from dendrite.core.expression import Expression as E
from dendrite.mathematics.elementary import sqrt
from dendrite.mathematics.trigonometry import arctan, arccos
from dendrite.mathematics.metric import norm

@E
def cylindrical() -> T:
  fx = arctan(y/x)
  fy = z
  fz = sqrt(x**2 + y**2)
  return (fx, fy, fz)

@E
# http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.436.1302&rep=rep1&type=pdf
def generalized_cylindrical(directrix: tuple) -> TDT:
  # Calculate Frenet frame https://www.wikiwand.com/en/Moving_frame#/Moving_tangent_frames
  tangent = np.array([diff(d, t) for d in directrix])
  tangent /= norm(tangent)

  normal = np.array([diff(tan, t) for tan in tangent])

  if not np.any(normal):
    raise ValueError("Directrix %s must have non-vanishing curvature." % directrix)

  normal /= norm(normal)

  binormal = np.cross(tangent, normal)
  binormal /= norm(binormal)

  p = np.array([x,y,z])
  p_prime = p - directrix
  distance = norm(p_prime)

  normal_coord = normal.dot(p_prime)
  binormal_coord = binormal.dot(p_prime)
  theta = arctan(normal_coord / binormal_coord)

  arc_length = integrate(sqrt(sum([diff(d, t)**2 for d in directrix])), (t, 0, t))
  approximate_arc_length = arc_length.as_sum(10)
  return ((theta, approximate_arc_length, distance), distance)

@E
def spherical() -> T:
  r = sqrt(x**2 + y**2 + z**2)
  fz = r
  fy = arccos(z/r)
  fx = arctan(y/x)
  return (fx,fy,fz)
