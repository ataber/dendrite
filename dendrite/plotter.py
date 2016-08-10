import numpy as np
import tensorflow as tf
import mcubes
import importlib
import sys
from dendrite.calculation import *
from dendrite.core.geometry import *

def raytrace(Object):
  raytracer = Raytracer(Object)
  raytracer.run()

def export_to_obj(geometry, resolution, bounds):
  volumetric = Volumetric(geometry, resolution, bounds)
  rendered = volumetric.run()[0]
  vertices, triangles = mcubes.marching_cubes(rendered, 0)
  shifted = []
  step_size = [(bounds[1][i] - bounds[0][i]) / resolution[i] for i in [0,1,2]]
  for vertex in vertices:
    v_shifted = [((c * step_size[i]) + bounds[0][i]) for i, c in enumerate(vertex)]
    shifted.append(v_shifted)

  mcubes.export_obj(shifted, triangles, geometry.name+".obj")

def export_to_graph_def(geometry):
  graph = Graph(geometry)
  graph_def = graph.session.graph.as_graph_def()
  tf.train.write_graph(graph_def, "./", geometry.name + ".pbtxt", True)

def export_to_cli(geometry, resolution, bounds):
  volumetric = Volumetric(geometry, resolution, bounds)
  rendered = volumetric.run()[0]

  unit = 0.001 # 0.001mm
  with open(geometry.name+".cli", "w") as file:
    file.write("$$HEADERSTART\n")
    file.write("$$ASCII\n")
    file.write("$$UNITS/"+str(unit)+"\n")
    file.write("$$HEADEREND\n")
    file.write("$$GEOMETRYSTART\n")

    slices = []
    for i in range(rendered.shape[-1]):
      contours = measure.find_contours(rendered[:,:,i], 0, fully_connected="high")
      shifted = []
      step_size = [(bounds[1][i] - bounds[0][i]) / resolution[i] for i in [0,1]]
      for connected_component in contours:
        shifted_component = []
        for vertex in connected_component:
          v_shifted = [((c * step_size[i]) - bounds[1][i]) for i, c in enumerate(vertex)]
          shifted_component.append(v_shifted)
        shifted.append(shifted_component)

      slices.append(shifted)

    start_height = bounds[0][2]
    end_height = bounds[1][2]
    step_size = (end_height - start_height) / len(slices)

    for i, _slice in enumerate(slices):
      z_height = start_height + (i*step_size)
      file.write("$$LAYER/"+str(z_height)+"\n")
      for j, connected_component in enumerate(_slice):
        file.write("$$POLYLINE/"+str(j)+",1,"+str(len(connected_component)))
        for line in connected_component:
          file.write(","+",".join([str(point) for point in line]))
        file.write("\n")

    file.write("$$GEOMETRYEND\n")

if __name__ == "__main__":
  object_name = sys.argv[1]
  Object = importlib.import_module("Parts."+object_name)
  export_to_obj(Object.geometry, Object.default_resolution, Object.default_bounds)
  # raytrace(Object.geometry)
