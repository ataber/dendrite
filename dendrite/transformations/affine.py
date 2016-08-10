import numpy as np
from sympy.abc import x,y,z
from dendrite.mathematics.linear_algebra import axis_angle_matrix, reflection_matrix
from dendrite.core.transformation import Transformation as T
from dendrite.decorators.type_coercion import coerce_types

@coerce_types
def identity() -> T:
  return (x,y,z)

@coerce_types
def translate(offset: np.ndarray) -> T:
  a, b, c = offset
  return (x-a, y-b, z-c)

@coerce_types
def rotate(axis: np.ndarray, radians: float) -> T:
  return tuple(np.dot(axis_angle_matrix(axis, radians), [x,y,z]))

@coerce_types
def reflect(axis: np.ndarray) -> T:
  return tuple(np.dot(reflection_matrix(axis), [x,y,z]))
