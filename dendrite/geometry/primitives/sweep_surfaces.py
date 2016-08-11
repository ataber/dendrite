import sympy
from sympy.abc import x,y,z,t
from dendrite.core.functional import Functional as F
from dendrite.core.expression import Expression as E
from dendrite.geometry.primitives.quadrics import sphere
from dendrite.decorators.type_coercion import coerce_types

def eliminate_variable(dependent, variable):
  return sympy.resultant(dependent, sympy.diff(dependent, variable), variable)

@coerce_types
def channel_surface(radius, directrix) -> F:
  distance_to_curve = [coordinate - curve for coordinate, curve in zip([x,y,z], directrix)]
  time_dependent = sphere(radius)(*distance_to_curve)
  return eliminate_variable(time_dependent, t)
