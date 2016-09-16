import numpy as np
from sympy.abc import x,y,z
from dendrite.mathematics.linear_algebra import axis_angle_matrix, reflection_matrix
from dendrite.core.transformation import Transformation as T
from dendrite.core.expression import Expression as E
from dendrite.decorators.type_coercion import three_vector

@E
def translate(a, b, c) -> T:
  return (x-a, y-b, z-c)

@E
def rotate(axis: three_vector, radians) -> T:
  return tuple(np.dot(axis_angle_matrix(axis, radians), [x,y,z]))

@E
def reflect(axis: three_vector) -> T:
  return tuple(np.dot(reflection_matrix(axis), [x,y,z]))
