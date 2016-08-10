from flask import Flask, jsonify, request
from flask.ext.cors import CORS
import mcubes
import json
import sys
import importlib
from Calculation.Graph import Graph
from Calculation.Volumetric import Volumetric

app = Flask(__name__)
CORS(app)

@app.route("/")
def render():
  request_params = {}
  for param_name, default_value in params.items():
    request_params[param_name] = float(request.args.get(param_name, default_value))

  geometry.set_parameters(request_params)

  rendered, volume, com, inertia = volumetric.run(
    bounds=bounds,
    op_names=["draw", "volume", "com", "inertia"],
  )

  infoArr = zip(['volume', "centerOfMass", "inertia"], [float(volume), com.tolist(), inertia.tolist()], ["float", "vec3", "vec3"])
  objectInfoStrings = [
    {
      "text": name,
      "type": t,
      "value": value
    } for name, value, t in infoArr
  ]

  js = {
    "objectInfoStrings": objectInfoStrings,
    "resolution": [int(r) for r in resolution],
    "boundsMin": [float(b) for b in bounds[0]],
    "boundsMax": [float(b) for b in bounds[1]]
  }

  with open("volumetric_data.json", "w") as f:
    f.write(json.dumps(js))

  if request.args.get("output", "triangles") == "triangles":
    vertices, triangles = mcubes.marching_cubes(rendered, 0)
    shifted = []
    step_size = [(bounds[1][i] - bounds[0][i]) / resolution[i] for i in [0,1,2]]
    for vertex in vertices:
      v_shifted = [((c * step_size[i]) + bounds[0][i]) for i, c in enumerate(vertex)]
      shifted.append(v_shifted)

    js["vertices"] = shifted
    js["triangles"] = triangles.tolist()
  elif request.args.get("output", "distanceField") == "distanceField":
    js["distanceFieldData"] = rendered.tolist()

  return jsonify(js)

if __name__ == "__main__":
  object_name = sys.argv[1]
  Object = importlib.import_module("demo_parts."+object_name)
  params = Object.default_parameters
  objectVariables = [
    {
      "text": name,
      "minValue": Object.param_ranges[name][0],
      "maxValue": Object.param_ranges[name][1],
      "steps": 100,
      "defaultStep": int((default - Object.param_ranges[name][0]) // ((Object.param_ranges[name][1] - Object.param_ranges[name][0]) / 100))
    } for name, default in Object.default_parameters.items()
  ]
  with open("objectVariables.json", "w") as f:
    f.write(json.dumps({"objectVariables": objectVariables}))

  bounds = Object.default_bounds
  resolution = Object.default_resolution
  geometry = Object.geometry
  volumetric = Volumetric(geometry, resolution, bounds)

  app.debug = True
  app.run(host='0.0.0.0')
