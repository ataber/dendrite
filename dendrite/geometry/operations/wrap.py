import sympy
from sympy.abc import t
from dendrite.core.functional import Functional as F
from dendrite.core.expression import Expression as E
from dendrite.transformations.coordinate import generalized_cylindrical
from dendrite.geometry.primitives.linear import plane
from dendrite.transformations.affine import translate

@E
def pattern_along_curve(obj: F, directrix: tuple, bounds: tuple = (0, 1)) -> F:
  curve_transform = generalized_cylindrical(directrix, bounds)
  patterned_obj = obj << curve_transform
  tangent = [sympy.diff(d, t) for d in directrix]

  bounding_planes = []
  for b in bounds:
    endpoint_tangent = [tan.subs({t: b}) for tan in tangent]
    endpoint = [d.subs({t: b}) for d in directrix]
    bounding_plane = plane(endpoint_tangent, 0) << translate(*endpoint)
    bounding_planes.append(bounding_plane)

  scoped = patterned_obj // ~bounding_planes[0] // bounding_planes[1]
  return scoped
