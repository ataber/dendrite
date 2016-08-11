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
    space = tf.Variable(space_grid.astype(np.float32), name="Space", trainable=False)

    # Name TF ops for better graph visualization
    x = tf.squeeze(tf.slice(space, [0,0,0,0], [1,-1,-1,-1]), squeeze_dims=[0], name="X-Coordinates")
    y = tf.squeeze(tf.slice(space, [1,0,0,0], [1,-1,-1,-1]), squeeze_dims=[0], name="Y-Coordinates")
    z = tf.squeeze(tf.slice(space, [2,0,0,0], [1,-1,-1,-1]), squeeze_dims=[0], name="Z-Coordinates")

    draw_op = functional(x,y,z)

    with tf.name_scope("Inside"):
      dim_lengths = [
        (x[coordinate_shape[0]-1,0,0] - x[0,0,0]) / coordinate_shape[0],
        (y[0,coordinate_shape[1]-1,0] - y[0,0,0]) / coordinate_shape[1],
        (z[0,0,coordinate_shape[2]-1] - z[0,0,0]) / coordinate_shape[2],
      ]
      voxel_volume = dim_lengths[0] * dim_lengths[1] * dim_lengths[2]
      inside = tf.sign(tf.maximum(draw_op, tf.zeros_like(draw_op)))

    with tf.name_scope("CalculateVolume"):
      volume_op = tf.reduce_sum(inside) * voxel_volume

    with tf.name_scope("COM"):
      x_com = tf.reduce_sum(inside * x * voxel_volume) / volume_op
      y_com = tf.reduce_sum(inside * y * voxel_volume) / volume_op
      z_com = tf.reduce_sum(inside * z * voxel_volume) / volume_op
      com = tf.pack([x_com, y_com, z_com])

    with tf.name_scope("IntertialMoment"):
      x_cm = x + x_com
      y_cm = y + y_com
      z_cm = z + z_com
      i_xx = tf.reduce_sum(inside * (tf.square(y_cm) + tf.square(z_cm)) * voxel_volume)
      i_yy = tf.reduce_sum(inside * (tf.square(x_cm) + tf.square(z_cm)) * voxel_volume)
      i_zz = tf.reduce_sum(inside * (tf.square(x_cm) + tf.square(y_cm)) * voxel_volume)
      inertia = tf.pack([i_xx, i_yy, i_zz])

    graph = Graph(geometry, debug=debug)
    graph.run(tf.initialize_all_variables())
    graph.session.graph.finalize()

    self.graph = graph
    self.placeholders = placeholders
    self.coordinate_shape = coordinate_shape
    self.ops = {
      "draw": draw_op,
      "volume": volume_op,
      "com": com,
      "inertia": inertia
    }

  def run(self, bounds=[[-1,-1,-1],[1,1,1]], op_names=["draw"]):
    print("Running: Volumetric")
    ops = [self.ops[op_name] for op_name in op_names]
    with self.graph.tensorboard_logging("Volumetric"):
      return self.graph.run(ops, {})
