import numpy as np
from dendrite.mathematics.trigonometry import cos, sin
from dendrite.mathematics.elementary import square

def axis_angle_matrix(axis, angle):
  axis = axis / np.linalg.norm(axis)
  ux, uy, uz = axis
  c = cos(angle)
  s = sin(angle)
  return np.array([
    [c+(square(ux)*(1-c)), (ux*uy*(1-c)-uz*s), (ux*uz*(1-c)+uy*s)],
    [(ux*uy*(1-c)+uz*s), c+(square(uy)*(1-c)), (uy*uz*(1-c)-ux*s)],
    [(ux*uz*(1-c)-uy*s), (uy*uz*(1-c)+ux*s), c+(square(uz)*(1-c))]
  ])

def reflection_matrix(axis):
  axis = axis / np.linalg.norm(axis)
  ux, uy, uz = axis
  return np.array([
    [(1-2*ux**2), (-2*ux*uy), (-2*ux*uz)],
    [(-2*ux*uy), (1-2*uy**2), (-2*uy*uz)],
    [(-2*ux*uz), (-2*uy*uz), (1-2*uz**2)]
  ])
