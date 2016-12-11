from sympy import diff, integrate, Integral
from dendrite.mathematics.elementary import sqrt
from sympy.abc import t

def approximate_arc_length(curve, sum_terms=20):
  arc_length = Integral(sqrt(sum([diff(d, t)**2 for d in curve])), (t, 0, t))
  return arc_length.as_sum(sum_terms)
