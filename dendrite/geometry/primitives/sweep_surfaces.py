import sympy
from sympy.abc import x,y,z,t
from dendrite.core.functional import Functional as F
from dendrite.core.expression import Expression as E
from dendrite.geometry.primitives.quadrics import sphere_squared as sphere
from dendrite.decorators.type_coercion import coerce_output

def eliminate_variable(dependent, variable):
  try:
    independent = sympy.resultant(dependent, sympy.diff(dependent, variable), variable)
  except sympy.polys.polyerrors.PolynomialDivisionFailed as e:
    print("Unable to eliminate %s from %s" % (variable, dependent))
    raise(e)
  if independent == 0:
    raise ValueError("Unable to eliminate %s from %s: is the polynomial time-dependent?" % (variable, dependent))
  else:
    return sympy.factor(independent)

@coerce_output
def channel_surface(radius, directrix: tuple) -> F:
  distance_to_curve = [coordinate - curve for coordinate, curve in zip([x,y,z], directrix)]
  time_dependent = sphere(radius)(*distance_to_curve)
  return eliminate_variable(time_dependent, t)

@coerce_output
def sweep_surface(shape: F, directrix: tuple) -> F:
  distance_to_curve = [coordinate - curve for coordinate, curve in zip([x,y,z], directrix)]
  time_dependent = shape(*distance_to_curve)
  return eliminate_variable(time_dependent, t)
