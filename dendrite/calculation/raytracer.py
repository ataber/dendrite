import numpy as np
import tensorflow as tf
from dendrite.mathematics.metric import *
from dendrite.core.functional import Functional
from dendrite.core.geometry import Geometry
from dendrite.calculation.graph import Graph

def vector_fill(shape, vector):
  if isinstance(vector, tf.Tensor):
    with tf.name_scope("VectorFill"):
      return tf.pack([
        tf.fill(shape, vector[0]),
        tf.fill(shape, vector[1]),
        tf.fill(shape, vector[2]),
      ])
  else:
    return np.array([
      np.tile(vector[0], shape),
      np.tile(vector[1], shape),
      np.tile(vector[2], shape)
    ])

def normalize_vector(vector):
  with tf.name_scope("Normalize"):
    return vector / norm(vector)

def colormix(a,b,factor=0.5):
  with tf.name_scope("Colormix"):
    return a*(1-factor) + b*(factor)

class Raytracer:
  def __init__(self, geometry, resolution=(1920,1080), debug=False):
    if isinstance(geometry, Geometry):
      functional = geometry.functional
    elif isinstance(geometry, Functional):
      functional = geometry
    else:
      raise ValueError("Can't evaluate instance of %s" % type(geometry))

    debug_x, debug_y = (240,108)
    placeholders = {}

    with tf.name_scope("Camera"):
      # First coordinates in u,v form
      aspect_ratio = resolution[0]/resolution[1]
      min_bounds, max_bounds = (-aspect_ratio, -1), (aspect_ratio, 1)
      resolutions = list(map(lambda x: x*1j, resolution))
      image_plane_coords = np.mgrid[min_bounds[0]:max_bounds[0]:resolutions[0],min_bounds[1]:max_bounds[1]:resolutions[1]]

      # Find the center of the image plane
      camera_position = tf.placeholder(dtype=tf.float32, shape=(3,), name="CameraPosition")
      placeholders["camera_position"] = camera_position

      lookAt = (0,0,-0.1)
      camera = camera_position - np.array(lookAt)
      camera_direction = normalize_vector(camera)
      focal_length = 1
      eye = camera + focal_length * camera_direction

      # Coerce into correct shape
      image_plane_center = vector_fill(resolution, camera_position)

      # Convert u,v parameters to x,y,z coordinates for the image plane
      v_unit = [0,0,-1]
      u_unit = tf.cross(camera_direction, v_unit)
      image_plane = image_plane_center + image_plane_coords[0] * vector_fill(resolution, u_unit) + image_plane_coords[1] * vector_fill(resolution, v_unit)

      # Populate the image plane with initial unit ray vectors
      initial_vectors = image_plane - vector_fill(resolution, eye)
      ray_vectors = normalize_vector(initial_vectors)

    with tf.name_scope("SetupSpace"):
      # t is the length along each ray
      t = tf.Variable(tf.zeros_initializer(resolution, dtype=tf.float32), name="ScalingFactor")
      reset = t.assign(tf.zeros(resolution, dtype=tf.float32))
      tf.scalar_summary("Debug Vector", t[debug_x, debug_y])

      space = (ray_vectors * t) + image_plane
      space = tf.Print(space, [space[:, debug_x, debug_y]], message="space")

      # Name TF ops for better graph visualization
      x = tf.squeeze(tf.slice(space, [0,0,0], [1,-1,-1]), squeeze_dims=[0], name="X-Coordinates")
      y = tf.squeeze(tf.slice(space, [1,0,0], [1,-1,-1]), squeeze_dims=[0], name="Y-Coordinates")
      z = tf.squeeze(tf.slice(space, [2,0,0], [1,-1,-1]), squeeze_dims=[0], name="Z-Coordinates")

    evaluated_functional = functional(x,y,z)
    tf.scalar_summary("Functional", evaluated_functional[debug_x, debug_y])
    evaluated_functional = tf.Print(evaluated_functional, [evaluated_functional[debug_x, debug_y]], message="Functional")

    with tf.name_scope("Lighting"):
      light = {"position": np.array([0,1,1]), "color": np.array([255,255,255])}
      gradient = tf.pack(tf.gradients(evaluated_functional, [x,y,z]))
      gradient = tf.Print(gradient, [gradient[:,debug_x, debug_y]], message="Grad")

      with tf.name_scope("CalculateIncidence"):
        # Note this is strictly the negative normal vector, as grad(F) points inward
        normal_vector = normalize_vector(gradient)
        incidence = normal_vector - vector_fill(resolution, light["position"])
        normalized_incidence = normalize_vector(incidence)
        incidence_angle = tf.reduce_sum(normalized_incidence * normal_vector, reduction_indices=0)

        # Remove incidence angles greater than 90 degrees, those correspond to occluded surfaces
        incidence_angle = tf.maximum(incidence_angle, tf.zeros_like(incidence_angle))

      with tf.name_scope("AddLight"):
        # Split the color into three channels
        light_intensity = vector_fill(resolution, light['color']) * incidence_angle

        # Add ambient light + fog
        ambient_light = {"color": np.array([119, 139, 165])}
        sky_color = [70, 130, 180]
        with_ambient = colormix(light_intensity, vector_fill(resolution, ambient_light["color"]))

        clip_length = 100
        with_fog = colormix(with_ambient, vector_fill(resolution, sky_color), (1-tf.exp(-t/clip_length)))
        lighted = with_ambient

      with tf.name_scope("BooleanMask"):
        # Mask out pixels not on surface
        epsilon = 0.0001
        bitmask = tf.less_equal(tf.abs(evaluated_functional), epsilon)
        masked = lighted * tf.to_float(bitmask)
        background = vector_fill(resolution, sky_color) * tf.to_float(tf.logical_not(bitmask))
        image_data = tf.cast(masked + background, tf.uint8)

    with tf.name_scope("RayCasting"):
      # Criteria for convergence
      distance = tf.abs(evaluated_functional)
      converged = tf.logical_or(tf.greater(distance, clip_length), tf.less(distance, epsilon))
      tf.scalar_summary("Not Converged", tf.reduce_sum(tf.to_int32(tf.logical_not(converged))))

      minimum_step = epsilon
      distance_step = t - (tf.sign(evaluated_functional) * tf.maximum(distance, minimum_step))
      distance_step = tf.Print(distance_step, [distance_step[debug_x, debug_y]], message="dist")
      cast_op = t.assign(tf.select(
        converged,
        t,
        distance_step
      ))

    with tf.name_scope("Render"):
      image = tf.transpose(image_data)
      render = tf.image.encode_jpeg(image)

      # debug_quantity = tf.pack((tf.to_int32(converged)*255,)*3)
      # debug_quantity = gradient
      debug_quantity = tf.pack((1/distance,)*3)
      image_data = tf.cast(tf.clip_by_value(debug_quantity, 0, 255), tf.uint8)
      debug_image = tf.transpose(image_data)
      debug_render = tf.image.encode_jpeg(debug_image)

    summaries = tf.merge_all_summaries()

    self.graph = Graph(geometry, debug=debug)
    self.placeholders = placeholders
    self.resolution = resolution
    self.ops = {
      "cast": cast_op,
      "render": render,
      "reset": reset,
      "debug_render": debug_render,
      "summaries": summaries,
      "converged_mask": converged
    }

  def run(self, frames=1, folder="output"):
    print("Running: Raytracer")
    self.graph.run(tf.initialize_all_variables())
    self.graph.session.graph.finalize()

    for frame in range(frames):
      print("Frame: " + str(frame))
      with self.graph.tensorboard_logging("frame: "+str(frame)+", rayupdate: "):
        image = self.render_frame(frame, frames)

      with open(folder+"/Raytraced"+str(frame)+".jpg", "wb") as f:
        f.write(image)

  def render_frame(self, frame, frames, max_steps=50):
    feed_dict = {}
    camera = [-0.9,-0.1,-0.1]
    feed_dict[self.placeholders["camera_position"]] = camera

    self.graph.run(self.ops['reset'])
    step = 0
    while step < max_steps:
      print("Render Step: " + str(step))
      _, debug, converged = self.graph.run(
        [self.ops['cast'], self.ops["debug_render"], self.ops['converged_mask']],
        feed_dict,
      )
      if np.sum(np.logical_not(converged)) < 20:
        break

      if self.graph.debug:
        with open("output/debug"+str(step)+"frame"+str(frame)+".jpg", "wb") as f:
          f.write(debug)

      self.graph.run_summary(self.ops["summaries"], feed_dict)
      step += 1

    return self.graph.run(self.ops['render'], feed_dict)
