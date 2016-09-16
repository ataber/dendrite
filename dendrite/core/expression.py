import inspect
import sympy
from sympy.abc import x,y,z
import numpy as np
from functools import partial
from collections import OrderedDict
from dendrite.decorators.type_coercion import *
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
        raise ValueError(self.function.__name__+": No value provided for " +name)

    def to_argument(s):
      annotation = self.parameters[s].annotation

      # These types should not be symbolized
      pass_through_types = (list, str, int, float, tuple, Operad)

      if annotation is three_vector:
        return sympy.MatrixSymbol(s, 3, 1)
      elif annotation is functional_lambda:
        return sympy.Lambda([x, y, z], 'f')
      elif annotation is transformation_lambda:
        return sympy.Lambda([x, y, z], ('gx', 'gy', 'gz'))
      elif issubclass(annotation, pass_through_types):
        return substitute_dict[s]
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
      symbols_dict = {
        k.name: substitute_dict[k.name]
        for k in symbols if isinstance(k, sympy.Symbol)
      }
      matrix_symbols_dict = {
        k.name: np.reshape(substitute_dict[k.name], k.shape)
        for k in symbols if isinstance(k, sympy.MatrixSymbol)
      }
      input_dict = {**symbols_dict, **matrix_symbols_dict}

    operad.set_inputs(**input_dict)
    return operad
