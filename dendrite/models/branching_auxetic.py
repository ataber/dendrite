from dendrite.models.auxetic_tiling import *
from dendrite.geometry.operations.boolean import *
from dendrite.geometry.primitives.sweep_surfaces import *

bounds = np.array([2,2,2])
default_bounds = [-bounds, bounds]
default_resolution = (150,)*3

def branch(shape=cut_plane(auxetic, 3)):
  endpoints = [[0,0,0], [0,4,4]]
  tangents = [[0,0,1], [1,1,3]]
  # cylinder = shape << cylindrical()
  main_e = [[0,0,-4], [1,1,3]]
  main_t = [[0,0,1], [1,2,1]]
  main_curve = hermite_interpolate_3d(*main_e, *main_t)
  main_branch = pattern_along_curve(shape, main_curve)

  # p = ~plane([0,0,1],0) << translate(0,0,0.4)
  # remove_cylinder = p << cylindrical()
  remove_main_branch = channel_surface(0.4, main_curve)
  curve = hermite_interpolate_3d(*endpoints, *tangents)
  # remove_intersect = pattern_along_curve(p, curve)
  remove_intersect = channel_surface(0.4, curve)
  branch = pattern_along_curve(shape, curve)
  return scale(polynomial_smooth_union((main_branch // remove_intersect), (branch // remove_main_branch), 0.1), 0.3)
  # return scale((branch // remove_cylinder), 0.3)

@G
def model():
  return branch()
