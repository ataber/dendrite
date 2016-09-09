from typing import List
from numpy import polynomial
from sympy import Rational
from sympy.abc import t
from dendrite.mathematics.elementary import *
from dendrite.decorators.type_coercion import coerce_types, rational_vector

# from http://paulbourke.net/miscellaneous/interpolation/
def hermite_interpolate(x1, x2, d1, d2):
  a0 = 2*t**3 - 3*t**2 + 1
  a1 = t**3 - 2*t**2 + t
  a2 = t**3 - t**2
  a3 = -2*t**3 + 3*t**2
  return a0*x1 + a1*d1 + a2*x2 + a2*d2

@coerce_types
def hermite_interpolate_3d(a: rational_vector, b: rational_vector, da: rational_vector, db: rational_vector):
  return tuple([hermite_interpolate(c1, c2, d1, d2) for c1, c2, d1, d2 in zip(a, b, da, db)])

def bernstein(degree, i):
  coefficients = [0,]*i + [choose(degree, i)]
  first_term = polynomial.polynomial.Polynomial(coefficients)
  second_term = polynomial.polynomial.Polynomial([1,-1])**(degree - i)
  return first_term * second_term

def bezier_curve(points):
  degree = len(points)
  curve = polynomial.polynomial.Polynomial([0])
  for i in range(degree):
    curve += Bernstein(degree, i) * points[i]
  return curve
