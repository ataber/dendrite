from numpy import polynomial
from sympy import Tuple, sympify, diff, integrate
from sympy.abc import t
from dendrite.mathematics.elementary import *
from dendrite.decorators.type_coercion import coerce_types, rational_vector

# from http://paulbourke.net/miscellaneous/interpolation/
# https://www.wikiwand.com/en/Cubic_Hermite_spline
def hermite_interpolate(p0, p1, m0, m1):
  return (2*t**3 - 3*t**2 + 1)*p0 + (t**3 - 2*t**2 + t)*m0 + (-2*t**3 + 3*t**2)*p1 + (t**3 - t**2)*m1

@coerce_types
def hermite_interpolate_3d(a: rational_vector, b: rational_vector, da: rational_vector, db: rational_vector):
  return tuple([hermite_interpolate(p0, p1, d0, d1) for p0, p1, d0, d1 in zip(a, b, da, db)])

def parametrize_by_arc_length(curve, bounds=(0,1)):
  integrand = sqrt(sum([diff(expr, t)**2 for expr in curve]))
  integrated = integrate(integrand, (t, *bounds)).evalf()
  return tuple([sympify(c).subs({t: t/integrated}) for c in curve])

def bernstein(degree, i):
  first_term = t**i
  second_term = (1-t)**(degree - i)
  return choose(degree, i) * first_term * second_term

def bezier_curve(points):
  # points is an array of [b(t)] where b is the function the curve interpolates
  curve = sympify(0)
  for i in range(len(points)):
    curve += bernstein(len(points) - 1, i) * points[i]
  return curve
