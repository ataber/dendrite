from sympy.abc import t
from dendrite.core.geometry import Geometry as G
from dendrite.geometry.primitives.sweep_surfaces import *
from dendrite.geometry.primitives.polygons import *
from dendrite.transformations.affine import *
from dendrite.geometry.operations.morph import scale

default_resolution = [150,150,150]
default_bounds = [[-1,-1,-1],[1,1,1]]

@G
def model():
  endpoints = [[-1,0,0], [1,1,0]]
  tangents = [[2,0,0], [2,1,0]]

  directrix = hermite_interpolate_3d(*endpoints, *tangents)

  return distance_surface(0.5, directrix)
  # return scale(pipe(0.5, 0.1, endpoints, tangents), 0.5)
  # return scale(scoped_canal(0.5, endpoints, tangents), 0.5)
  # return scale(interpolating_canal(0.5, endpoints, tangents), 0.5)
