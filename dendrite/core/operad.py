import inspect
import sympy
import tensorflow as tf
from sympy.abc import x, y, z, t
from numbers import Number
from collections import OrderedDict
from dendrite.codegen.glsl_codegen import glslcodegen

class Operad:
  def __init__(self, expression, namespace=None, to_solve=None):
    self.expression = expression
    self._to_solve = to_solve
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
      self._symbolic_lambda = sympy.lambdify([x,y,z], self.substitute_symbols(), "sympy")
    return self._symbolic_lambda

  def substitute_symbols(self):
    subs = {}

    if type(self.expression) == sympy.Subs:
      gx, gy, gz = self.inputs["g"]()
      f = self.inputs["f"]
      return f(gx, gy, gz)
    else:
      for name, value in self.inputs.items():
        if isinstance(value, Operad):
          subs[name] = value()
        else:
          subs[name] = value

    return substitute(self.expression, subs)

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

          # expression is time-dependent, must solve for t
          if self._to_solve is not None:
            condition = lambda i, _: i < 20
            def body(i, _t):
              input_tensors[t] = _t
              variable_list = list(input_tensors.keys())
              to_solve = sympy.lambdify(variable_list, self._to_solve, "tensorflow")
              residual = to_solve(*input_tensors.values())
              residual = tf.Print(residual, [residual[45:50,45:50,45:50]])
              gradient = tf.gradients(residual, [_t])[0]
              gradient = tf.Print(gradient, [gradient[45:50,45:50,45:50]])
              return (i+1, _t - (residual / gradient))

            time = tf.zeros(X.get_shape(), name="Time_"+self.namespace)
            _, solution = tf.while_loop(
              condition,
              body,
              [tf.constant(0), time])
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
