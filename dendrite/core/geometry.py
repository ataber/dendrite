import tensorflow as tf
import inspect
from numbers import Number

class Geometry:
  def __init__(self, function, expose_parameters=True):
    self.function = function
    self.name = function.__name__
    self.expose_parameters = expose_parameters

    self.sub_parts = {}
    for k, v in inspect.getclosurevars(function).globals.items():
      if isinstance(v, Geometry):
        v.expose_parameters = False
        self.sub_parts[k] = v

    self.parameters = {}
    for name, param in inspect.signature(self.function).parameters.items():
      if param.default is inspect.Parameter.empty:
        raise ValueError("All Geometry parameters must have default values. Offending param: " + name)
      else:
        self.parameters[name] = param.default

  def __call__(self, **kwargs):
    for k, v in kwargs.items():
      self.parameters[k] = v
    return self.functional

  @property
  def functional(self):
    self.placeholders = {}
    arguments = {}
    for name, param in self.parameters.items():
      if self.expose_parameters and isinstance(param, Number):
        placeholder = tf.placeholder(tf.float32, shape=(), name=name)
        self.placeholders[name] = placeholder
        arguments[name] = placeholder
      else:
        arguments[name] = param

    functional = self.function(**arguments)
    functional.functional_namespace = self.name.title()
    return functional
