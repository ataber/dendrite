from sympy import Symbol
from dendrite.core.functional import Functional as F
from dendrite.core.expression import Expression as E
from dendrite.mathematics.elementary import bound
from dendrite.transformations.domain import scale_inputs
from dendrite.decorators.type_coercion import coerce_types

@coerce_types
def scale(a: F, s) -> F:
  return (a << scale_inputs(*(s,)*3)) * s

@E
def morph(a, b, t) -> F:
  t_bound = bound(t, 0, 1)
  return (a * t_bound) + (b * (1 - t_bound))
