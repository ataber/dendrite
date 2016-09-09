import inspect
import sympy
import numpy as np
from functools import partial
from collections import OrderedDict
from dendrite.decorators.type_coercion import coerce_output
from dendrite.core.operad import Operad

class Expression:
  def __init__(self, function):
    self.function = function
    self.parameters = inspect.signature(self.function).parameters
    self.arguments = OrderedDict()
    # Note the order of this dictionary is necessary for lining up positional args at call time
    for name, param in self.parameters.items():
      self.arguments[name] = param.default

  def __call__(self, *args, **kwargs):
    arg_list = list(self.arguments.items())
    substitute_dict = self.arguments.copy()

    if len(args) > len(arg_list):
      raise ValueError("Too many args: " + str(args))

    for i, arg in enumerate(args):
      substitute_dict[arg_list[i][0]] = arg

    for name, arg in kwargs.items():
      substitute_dict[name] = arg

    for name, arg in substitute_dict.items():
      if arg is inspect.Parameter.empty:
        raise ValueError("No value provided for " +name)

    def to_argument(s):
      annotation = self.parameters[s].annotation

      # These types should not be symbolized
      pass_through_types = (list, str, int, float, tuple, Operad)

      type_converters = {
        # np.array is not a class, just a function for creating ndarrays
        np.ndarray: np.array,
      }

      if issubclass(annotation, sympy.Tuple):
        return sympy.Tuple(*sympy.symbols("gx gy gz"))
      elif issubclass(annotation, pass_through_types):
        return substitute_dict[s]
      elif annotation in type_converters:
        return type_converters[annotation](substitute_dict[s])
      else:
        return sympy.Symbol(s)

    symbols = [to_argument(s) for s in self.arguments]
    operad = coerce_output(self.function)(*symbols)

    if isinstance(operad.expression, sympy.Subs):
      input_dict = {
        "f": substitute_dict["f"],
        "g": substitute_dict["g"]
      }
    else:
      input_dict = {
        k.name: substitute_dict[k.name]
        for k in symbols if isinstance(k, sympy.Symbol)
      }

    operad.set_inputs(**input_dict)
    return operad
