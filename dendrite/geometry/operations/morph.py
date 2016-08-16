from dendrite.core.functional import Functional as F
from dendrite.core.expression import Expression as E
from dendrite.mathematics.elementary import bound
from dendrite.transformations.domain import scale_inputs

@E
def scale(a: F, s) -> F:
  return s * (a << scale_inputs(*(s,)*3))()

@E
def morph(a: F, b: F, t) -> F:
  t_bound = bound(t, 0, 1)
  return (a * t_bound) + (b * (1 - t_bound))
