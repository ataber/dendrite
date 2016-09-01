import inspect
import sympy
from sympy.abc import x,y,z
from numbers import Number
from tensorflow import name_scope, Tensor, convert_to_tensor, float32
from collections import OrderedDict

class Operad:
  def __init__(self, expression, namespace=None):
    self.expression = expression
    self._symbolic_lambda = None
    self.inputs = {}
    self.namespace = namespace
    self.functional_namespace = None

  def set_inputs(self, **inputs):
    self.inputs = {**inputs, **self.inputs}

  def __call__(self, X=x, Y=y, Z=z, scope=None, functional_scope=None):
    if any([isinstance(c, Tensor) for c in [X, Y, Z]]):
      return self.compute_tensorflow(X, Y, Z, scope=scope, functional_scope=functional_scope)
    return self.symbolic_lambda(X, Y, Z)

  @property
  def symbolic_lambda(self):
    if self._symbolic_lambda is None:
      self._symbolic_lambda = sympy.lambdify([x,y,z], self.substitute_symbols(), "sympy")
    return self._symbolic_lambda

  def substitute_symbols(self):
    subs = {}

    if type(self.expression) == sympy.Subs:
      gx, gy, gz = self.inputs["g"].substitute_symbols()
      f = self.inputs["f"]
      return f(gx, gy, gz)
    else:
      for name, value in self.inputs.items():
        if isinstance(value, Operad):
          subs[name] = value.substitute_symbols()
        else:
          subs[name] = value

    return substitute(self.expression, subs)

  def compute_tensorflow(self, X, Y, Z, scope=None, functional_scope=None):
    namespace = self.namespace or scope

    if self.functional_namespace and functional_scope:
      functional_namespace = functional_scope + self.functional_namespace
    else:
      functional_namespace = self.functional_namespace or functional_scope

    with name_scope(None):
      # Clear scope, otherwise functional components end up nested within Operations
      with name_scope(functional_namespace) as functional_scope:
        with name_scope(namespace) as scope:
          if type(self.expression) == sympy.Subs:
            gx, gy, gz = self.inputs["g"](X, Y, Z, scope=scope, functional_scope=functional_scope)
            f = self.inputs["f"]
            return f(gx, gy, gz, scope=scope, functional_scope=functional_scope)

          input_tensors = OrderedDict()
          for name, value in self.inputs.items():
            if isinstance(value, Operad):
              input_tensors[name] = value(X, Y, Z, scope=scope, functional_scope=functional_scope)
            elif isinstance(value, sympy.Basic):
              input_tensors[name] = sympy.lambdify([x, y, z], value, "tensorflow")(X, Y, Z)
            else:
              input_tensors[name] = convert_to_tensor(value, dtype=float32)

          variable_list = [x, y, z] + list(input_tensors.keys())

          tf_lambda = sympy.lambdify(variable_list, self.expression, "tensorflow")
          return tf_lambda(X, Y, Z, *input_tensors.values())

  def __lshift__(self, other):
    raise ValueError("Can not compose types: %s and %s" % (type(self), type(other)))

  def __repr__(self):
    return "<Core.Operad."+type(self).__name__+": "+str(self.namespace)+"> {"+str(self.expression)+"}"

def substitute(expr, subs):
  if isinstance(expr, tuple):
    return sympy.Tuple(*[substitute(e, subs) for e in expr])
  substituted = sympy.lambdify(subs.keys(), expr, "sympy")(**subs)
  return substituted
