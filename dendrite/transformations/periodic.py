from sympy.abc import x,y,z
from dendrite.mathematics.elementary import floor
from dendrite.core.transformation import Transformation as T
from dendrite.transformations.domain import absolute_value
from dendrite.decorators.type_coercion import coerce_types

@coerce_types
def saw_wave(amplitudes, frequencies, D: str = "xyz") -> T:
  # amplitudes specify lengths of input unit cell, centered at origin
  # frequencies specify lengths of replica cell
  fx, fy, fz = [
    amplitude * ((coordinate / freq) - floor((coordinate / freq) + 1/2))
    for amplitude, freq, coordinate in zip(amplitudes, frequencies, [x, y, z])
  ]

  if 'x' not in D:
    fx = amplitudes[0] * (x / frequencies[0])
  if 'y' not in D:
    fy = amplitudes[1] * (y / frequencies[1])
  if 'z' not in D:
    fz = amplitudes[2] * (z / frequencies[2])
  return (fx,fy,fz)

@coerce_types
def triangle_wave(amplitudes, periods, D: str = "xyz") -> T:
  return absolute_value() << saw_wave(amplitudes, periods, D)
