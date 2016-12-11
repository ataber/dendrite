import sympy
import tensorflow as tf
from sympy.abc import x, y, z, t
from collections import OrderedDict
from dendrite.utils.tensorflow_utils import *
from dendrite.codegen.glsl_codegen import glslcodegen

class Operad:
  def __init__(self, expression, namespace=None, to_minimize=None, time_bounds=None):
    self.expression = expression
    self._to_minimize = to_minimize
    self._time_bounds = time_bounds
    self._symbolic_lambda = None
    self.inputs = {}
    self.namespace = namespace
    self.functional_namespace = None

  def set_inputs(self, **inputs):
    self.inputs = {**inputs, **self.inputs}

  def __call__(self, X=x, Y=y, Z=z, scope=None, functional_scope=None):
    if any([isinstance(c, (tf.Variable, tf.Tensor)) for c in [X, Y, Z]]):
      return self.compute_tensorflow(X, Y, Z, scope=scope, functional_scope=functional_scope)
    return self.symbolic_lambda(X, Y, Z)

  @property
  def symbolic_lambda(self):
    if self._symbolic_lambda is None:
      substituted = self.substitute_symbols()
      if isinstance(substituted, sympy.Subs):
        substituted = substituted.doit()
      self._symbolic_lambda = sympy.lambdify([x,y,z], substituted, "sympy")
    return self._symbolic_lambda

  def substitute_symbols(self):
    substitute_dict = {}

    if type(self.expression) == sympy.Subs:
      gx, gy, gz = self.inputs["g"].substitute_symbols()
      g0, g1, g2 = self.expression.args[2]
      f = self.inputs["f"].substitute_symbols()
      return self.expression.subs({"f": f, g0: gx, g1: gy, g2: gz}, simultaneous=True)
    else:
      for name, value in self.inputs.items():
        if isinstance(value, Operad):
          substitute_dict[name] = value.substitute_symbols()
        else:
          substitute_dict[name] = value

    def substitute(expr, subs):
      if isinstance(expr, tuple):
        return tuple([substitute(e, subs) for e in expr])
      # Use symbolic lambda instead of expr.subs; subs attempts to simplify at every step, causing much slow
      return sympy.lambdify(subs.keys(), expr, "sympy")(**subs)

    return substitute(self.expression, substitute_dict)

  def compute_tensorflow(self, X, Y, Z, scope=None, functional_scope=None):
    namespace = self.namespace or scope

    if self.functional_namespace and functional_scope:
      functional_namespace = functional_scope + self.functional_namespace
    else:
      functional_namespace = self.functional_namespace or functional_scope

    with tf.name_scope(None):
      # Clear scope, otherwise functional components end up nested within Operations
      with tf.name_scope(functional_namespace) as functional_scope:
        with tf.name_scope(namespace) as scope:
          if type(self.expression) == sympy.Subs:
            gx, gy, gz = self.inputs["g"](X, Y, Z, scope=scope, functional_scope=functional_scope)
            f = self.inputs["f"]
            return f(gx, gy, gz, scope=scope, functional_scope=functional_scope)

          input_tensors = OrderedDict()
          for sym, tens in zip([x,y,z], [X,Y,Z]):
            input_tensors[sym] = tens

          for name, value in self.inputs.items():
            if isinstance(value, Operad):
              input_tensors[name] = value(X, Y, Z, scope=scope, functional_scope=functional_scope)
            elif isinstance(value, sympy.Basic):
              input_tensors[name] = sympy.lambdify([x, y, z], value, "tensorflow")(X, Y, Z)
            else:
              input_tensors[name] = tf.convert_to_tensor(value, dtype=tf.float32)

          # expression is time-dependent, must minimize as a function of t
          if self._to_minimize is not None:
            if self._time_bounds is None:
              solution = newtons_method(self._to_minimize, input_tensors, X.get_shape())
            else:
              solution = bisection_method(self._to_minimize, input_tensors, X.get_shape(), self._time_bounds)
            input_tensors[t] = solution

          variable_list = list(input_tensors.keys())
          tf_lambda = sympy.lambdify(variable_list, self.expression, "tensorflow")
          return tf_lambda(*input_tensors.values())

  def __lshift__(self, other):
    raise ValueError("Can not compose types: %s and %s" % (type(self), type(other)))

  def __repr__(self):
    return "<Core.Operad."+type(self).__name__+": "+str(self.namespace)+"> {"+str(self.expression)+"}"

  def as_glsl(self):
    glsl_funcs = [d.as_glsl() if isinstance(d, Operad) else d for d in self.inputs.values()]
    glsl_funcs.append([glslcodegen((self.namespace.lower(), self.expression))])
    return glsl_funcs

def substitute(expr, subs):
  if isinstance(expr, tuple):
    return sympy.Tuple(*[substitute(e, subs) for e in expr])
  substituted = sympy.lambdify(subs.keys(), expr, "sympy")(**subs)
  return substituted
