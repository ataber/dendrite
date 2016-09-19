import numpy as np
from dendrite.core.geometry import Geometry as G
from dendrite.geometry.primitives.quadrics import *
from dendrite.geometry.primitives.polygons import *
from dendrite.geometry.primitives.linear import *
from dendrite.geometry.operations.morph import scale
from dendrite.geometry.operations.replication import replicate
from dendrite.geometry.operations.wrap import *
from dendrite.transformations.affine import *
from dendrite.transformations.coordinate import *
from dendrite.transformations.deformations import *
from dendrite.mathematics.polynomials import *
from dendrite.codegen.glsl_printer import *

bounds = np.array([2,2,2])
default_bounds = [-bounds, bounds]
default_resolution = (200,)*3

def polygon():
  return regular_polygon(0.4, 6) << rotate([0,0,1], np.pi/6)

def row(shape):
  return replicate(shape(), [1.6, 1, 3], [0.8, 0.5, 1], D="xy")

def micro(shape):
  row1 = row(shape=shape)
  row2 = row(shape=shape) << translate(0.4,0.25,0)
  return row1 | row2

def cut_plane(shape):
  m = micro(shape=shape)
  p = plane([0,0,1],0) // plane([0,0,1],-0.5)
  cut_p = p // m
  return cut_p << translate(0,0,0.4)

@G
def model():
  endpoints = [[0,2,0], [2,-1,0]]
  tangents = [[2,1,0], [1,-2,0]]
  directrix = hermite_interpolate_3d(*endpoints, *tangents)
  return scale(pattern_along_curve(cut_plane(polygon), directrix), 0.6)
  # return scale(cut_plane(polygon) << generalized_cylindrical(directrix), 0.8)
  # return scale((cut_plane(polygon) << cylindrical()) << deform(directrix), 0.5)
