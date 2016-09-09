import numpy as np
from sympy.abc import x,y,z
from dendrite.mathematics.linear_algebra import axis_angle_matrix, reflection_matrix
from dendrite.core.transformation import Transformation as T
from dendrite.core.expression import Expression as E

@E
def translate(offset: np.ndarray) -> T:
  a, b, c = offset
  return (x-a, y-b, z-c)

@E
def rotate(axis: np.ndarray, radians) -> T:
  return tuple(np.dot(axis_angle_matrix(axis, radians), [x,y,z]))

@E
def reflect(axis: np.ndarray) -> T:
  return tuple(np.dot(reflection_matrix(axis), [x,y,z]))
