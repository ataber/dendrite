from dendrite.mathematics.elementary import bound
from dendrite.core.functional import Functional as F
from dendrite.core.expression import Expression

# Following functions from http://www.iquilezles.org/www/articles/smin/smin.htm
# max = - min (-f,-g)
# IQ gives us the smooth intersect but we want union
@Expression
def polynomial_smooth_union(a, b, k=0.1) -> F:
  def polynomial_smooth_min(a, b, k, n=0.5):
    def mix(x, y, a) -> F:
      return -x*(a-1) + y*a
    h = (((b - a) / k) * n) + n
    return mix(b, a, h) + (h * (h - 1) * k)
  return -polynomial_smooth_min(-a, -b, k)

@Expression
def exponential_smooth_union(a, b, k=32) -> F:
  def exponential_smooth_min(a, b, k):
    h = exp(-a*k) + exp(-b*k)
    return -log(h) / k
  return -exponential_smooth_min(-a, -b, k)

@Expression
def power_smooth_union(a, b, k=8) -> F:
  def power_smooth_min(a, b, k):
    a = a ** k
    b = b ** k
    return ((a * b) / (a + b))**(1/k)
  return -power_smooth_min(-a, -b, k)
