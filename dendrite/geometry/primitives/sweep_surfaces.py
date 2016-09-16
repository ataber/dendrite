import sympy
from sympy.abc import x,y,z,t
from dendrite.core.functional import Functional as F
from dendrite.core.time_dependent_functional import TimeDependentFunctional as TDF
from dendrite.core.expression import Expression as E
from dendrite.geometry.primitives.quadrics import sphere_squared, sphere
from dendrite.geometry.primitives.linear import plane
from dendrite.transformations.affine import translate
from dendrite.decorators.type_coercion import coerce_output
from dendrite.mathematics.polynomials import *

def eliminate_variable(dependent, variable):
  try:
    # coerce coefficients to rational due to https://github.com/sympy/sympy/issues/11507
    dependent = sympy.sympify(str(dependent), rational=True)
    derivative = sympy.diff(dependent, variable)
    print("Computing resultant of "+str(dependent)+" and "+str(derivative))
    independent = sympy.resultant(dependent, derivative, variable)
  except sympy.polys.polyerrors.PolynomialDivisionFailed as e:
    print("Unable to eliminate %s from %s" % (variable, dependent))
    raise(e)
  if independent == 0:
    raise ValueError("Unable to eliminate %s from %s: is the polynomial time-dependent?" % (variable, dependent))
  else:
    result = factor(independent)
    print("Resultant: "+str(result))
    return result

def factor(eqn):
  factored = sympy.factor(eqn)
  if isinstance(factored.args[0], sympy.numbers.Number):
    numerical_factor = factored.args[0]
    # Preserve isosurface and orientation, remove potentially large numerical factors
    return sympy.sign(numerical_factor)*(factored/numerical_factor)
  return factored

@E
def channel_surface(radius: F, directrix: tuple) -> F:
  distance_to_curve = [coordinate - curve for coordinate, curve in zip([x,y,z], directrix)]
  # Use squared distance formula, to avoid radicals
  time_dependent = sphere_squared(radius)(*distance_to_curve)
  return eliminate_variable(time_dependent, t)

@E
def distance_surface(radius: F, directrix: tuple) -> TDF:
  distance_to_curve = [coordinate - curve for coordinate, curve in zip([x,y,z], directrix)]
  time_dependent = sphere(radius)(*distance_to_curve)
  diff_t = sympy.diff(time_dependent, t)
  return (time_dependent, time_dependent)

@E
def interpolating_canal(radius: F, endpoints: list, tangents: list) -> F:
  directrix = hermite_interpolate_3d(*endpoints, *tangents)
  return channel_surface(radius, directrix)

@E
def scoped_canal(radius: F, endpoints: list, tangents: list) -> F:
  channel = interpolating_canal(radius, endpoints, tangents)
  bounding_planes = [plane(tang, 0) << translate(*p) for p, tang in zip(endpoints, tangents)]
  return channel // ~bounding_planes[0] // bounding_planes[1]

@E
def pipe(radius: F, thickness: float, endpoints: list, tangents: list) -> F:
  inner = scoped_canal(radius - thickness, endpoints, tangents)
  outer = scoped_canal(radius, endpoints, tangents)
  return outer // inner
