import numpy as np
import tensorflow as tf
from dendrite.calculation.graph import Graph
from dendrite.core.functional import Functional
from dendrite.core.geometry import Geometry

class Volumetric:
  def __init__(self, geometry, coordinate_shape, bounds, debug=False):
    functional = geometry.functional

    placeholders = {}

    min_bounds, max_bounds = bounds
    resolutions = list(map(lambda x: x*1j, coordinate_shape))
    space_grid = np.mgrid[min_bounds[0]:max_bounds[0]:resolutions[0],min_bounds[1]:max_bounds[1]:resolutions[1],min_bounds[2]:max_bounds[2]:resolutions[2]]
    space_grid = space_grid.astype(np.float32)
    x = tf.Variable(space_grid[0,:,:,:], trainable=False, name="X-Coordinates")
    y = tf.Variable(space_grid[1,:,:,:], trainable=False, name="Y-Coordinates")
    z = tf.Variable(space_grid[2,:,:,:], trainable=False, name="Z-Coordinates")

    draw_op = functional(x,y,z)
    sanity_checked = tf.verify_tensor_all_finite(draw_op, "Sanity Check Failed", name="SanityCheck")

    graph = Graph(geometry, debug=debug)
    graph.run(tf.initialize_all_variables())
    graph.session.graph.finalize()

    self.graph = graph
    self.placeholders = placeholders
    self.coordinate_shape = coordinate_shape
    self.ops = {"draw": sanity_checked}

  def run(self, bounds=[[-1,-1,-1],[1,1,1]]):
    print("Running: Volumetric")
    with self.graph.tensorboard_logging("Volumetric"):
      return self.graph.run(self.ops["draw"], {})
