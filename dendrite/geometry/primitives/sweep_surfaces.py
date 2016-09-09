import sympy
from functools import reduce
from sympy.solvers.solveset import *
from sympy.abc import x,y,z,t
from dendrite.core.functional import Functional as F
from dendrite.core.expression import Expression as E
from dendrite.geometry.primitives.quadrics import sphere_squared as sphere
from dendrite.geometry.primitives.linear import plane
from dendrite.transformations.affine import translate
from dendrite.decorators.type_coercion import coerce_output
from dendrite.mathematics.polynomials import *

def eliminate_variable(dependent, variable):
  try:
    # coerce coefficients to rational due to https://github.com/sympy/sympy/issues/11507
    print("Computing resultant of "+str(dependent))
    dependent = sympy.sympify(str(dependent), rational=True)
    independent = sympy.resultant(dependent, sympy.diff(dependent, variable), variable)
  except sympy.polys.polyerrors.PolynomialDivisionFailed as e:
    print("Unable to eliminate %s from %s" % (variable, dependent))
    raise(e)
  if independent == 0:
    raise ValueError("Unable to eliminate %s from %s: is the polynomial time-dependent?" % (variable, dependent))
  else:
    return independent

@E
def channel_surface(radius: float, directrix: tuple) -> F:
  distance_to_curve = [coordinate - curve for coordinate, curve in zip([x,y,z], directrix)]
  time_dependent = sphere(radius)(*distance_to_curve)
  return eliminate_variable(time_dependent, t)

@E
def scoped_canal(radius: float, endpoints: list, tangents: list) -> F:
  directrix = hermite_interpolate_3d(*endpoints, *tangents)
  bounding_planes = [plane(*tang, 0) << translate(p) for p, tang in zip(endpoints, tangents)]
  channel = channel_surface(radius, directrix)
  # scoped = channel // bounding_planes[0] // bounding_planes[1]
  scoped = channel
  return scoped

@E
def pipe(radius: float, thickness: float, endpoints: list, tangents: list) -> F:
  inner = scoped_canal(radius - thickness, endpoints, tangents)
  outer = scoped_canal(radius, endpoints, tangents)
  return outer // inner
