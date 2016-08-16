import inspect
import sympy
import numpy as np
from functools import partial
from sympy.abc import x,y,z
from collections import OrderedDict
from dendrite.decorators.type_coercion import coerce_output
from dendrite.core.operad import Operad

class Expression:
  def __init__(self, function):
    self.arguments = OrderedDict()
    self.function = function
    self.parameters = inspect.signature(self.function).parameters
    for name, param in self.parameters.items():
      self.arguments[name] = param.default

  def __call__(self, *args, **kwargs):
    substitute_dict = self.arguments.copy()
    arg_list = list(self.arguments.items())

    if len(args) > len(arg_list):
      raise ValueError("Too many args: " + str(args))

    for i, arg in enumerate(args):
      substitute_dict[arg_list[i][0]] = arg

    for name, arg in kwargs.items():
      substitute_dict[self.arguments[name]] = arg

    for name, arg in substitute_dict.items():
      if arg is inspect.Parameter.empty:
        raise ValueError("No value provided for " +name)

    def to_argument(s):
      annotation = self.parameters[s].annotation
      pass_through_types = (list, str, int, tuple, Operad)
      type_converters = {
        np.ndarray: np.array,
        # This is a hack to ensure the coefficients of polynomials are rational for variable elimination
        float: lambda f: sympy.sympify(str(f), rational=True)
      }

      if issubclass(annotation, sympy.Tuple):
        return sympy.Tuple(*sympy.symbols("gx gy gz"))
      elif issubclass(annotation, pass_through_types):
        return substitute_dict[s]
      elif annotation in type_converters:
        return type_converters[annotation](substitute_dict[s])
      else:
        return sympy.Symbol(s)

    operad = coerce_output(self.function)(*[to_argument(s) for s in self.arguments])
    operad.set_inputs(**substitute_dict)
    return operad
