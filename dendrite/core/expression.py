import inspect
import sympy
from functools import partial
from sympy.abc import x,y,z
from collections import OrderedDict
from dendrite.decorators.type_coercion import coerce_output
from dendrite.core.operad import Operad

class Expression:
  def __init__(self, function):
    self.symbols = OrderedDict()
    self.function = function
    self.parameters = inspect.signature(self.function).parameters
    for name, param in self.parameters.items():
      self.symbols[name] = param.default

  def __call__(self, *args, **kwargs):
    substitute_dict = self.symbols.copy()

    for i, arg in enumerate(args):
      symbol_list = list(self.symbols.items())
      substitute_dict[symbol_list[i][0]] = arg

    for name, arg in kwargs.items():
      substitute_dict[self.symbols[name]] = arg

    for name, arg in substitute_dict.items():
      if arg is inspect.Parameter.empty:
        raise ValueError("No value provided for " +name)

    def symbol(s):
      if issubclass(self.parameters[s].annotation, sympy.Tuple):
        return sympy.Tuple(*sympy.symbols("gx gy gz"))
      elif issubclass(self.parameters[s].annotation, Operad):
        return substitute_dict[s]
      else:
        return sympy.Symbol(s)
    operad = coerce_output(self.function)(*[symbol(s) for s in self.symbols])
    operad.set_inputs(**substitute_dict)
    return operad
