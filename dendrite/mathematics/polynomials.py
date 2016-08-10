from numpy import polynomial
from Mathematics.Trigonometry import *
from Mathematics.Elementary import *

def Chebyshev(coefficients):
  return polynomial.chebyshev.Chebyshev(coefficients)

def Bernstein(degree, i):
  coefficients = [0,]*i + [choose(degree, i)]
  first_term = polynomial.polynomial.Polynomial(coefficients)
  second_term = polynomial.polynomial.Polynomial([1,-1])**(degree - i)
  return first_term * second_term

def BezierCurve(points):
  degree = len(points)
  curve = polynomial.polynomial.Polynomial([0])
  for i in range(degree):
    curve += Bernstein(degree, i) * points[i]
  return curve
