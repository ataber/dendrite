from numpy import pi
from sympy.abc import x,y,z
from functools import reduce
from dendrite.mathematics.elementary import Max
from dendrite.mathematics.trigonometry import cos, sin
from dendrite.core.functional import Functional as F
from dendrite.core.expression import Expression as E

# https://www.reddit.com/r/hypershape/comments/341bv4/what_are_some_implicit_equations_for_polygons_and/
@E
def regular_polygon(circumradius, n: int) -> F:
  return circumradius - reduce(Max, [x*cos(2*pi*j/n) + y*sin(2*pi*j/n) for j in range(n)])
