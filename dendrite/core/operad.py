import inspect
import sympy
from sympy.abc import x,y,z
from numbers import Number
from tensorflow import name_scope, Tensor
from collections import OrderedDict

class Operad:
  def __init__(self, expression, namespace=None):
    self.expression = expression
    self.lambda_expression = None
    self.inputs = {}
    self.computed = False
    self.namespace = namespace
    self.functional_namespace = None

  def set_inputs(self, **inputs):
    self.inputs = {**inputs, **self.inputs}

  def __call__(self, X=x, Y=y, Z=z, **kwargs):
    # if self.computed is False:
    #   namespace = self.namespace or kwargs.get("scope", None)
    #   if self.functional_namespace and kwargs.get("functional_scope", None):
    #     functional_namespace = kwargs["functional_scope"] + self.functional_namespace
    #     with name_scope(None):
    #       # Clear scope, otherwise functional components end up nested within Operations
    #       with name_scope(functional_namespace) as functional_scope:
    #         with name_scope(namespace):
    #           kwargs["functional_scope"] = functional_scope
    #           kwargs["scope"] = namespace
    #           self.computed = self.compute(*args)
    #   else:
    #     functional_namespace = self.functional_namespace or kwargs.get("functional_scope", None)
    #     with name_scope(functional_namespace) as functional_scope:
    #       with name_scope(namespace) as scope:
    #         kwargs["functional_scope"] = functional_scope
    #         kwargs["scope"] = scope
    #         self.computed = self.compute(*args)

    # if isinstance(self.computed, Tensor):
    #   return self.computed
    # else:
    #   # Allow for console debugging
    #   result = self.computed
    #   self.computed = False
    #   return result
    if any([isinstance(c, Tensor) for c in [X,Y,Z]]):
      return self.compute_tensorflow(X,Y,Z, **kwargs)
    elif any([isinstance(c, Number) for c in [X,Y,Z]]):
      return self.symbolic_lambda(X,Y,Z)
    else:
      return substitute(self.compute_symbolic(), {"x": X, "y": Y, "z": Z})

  @property
  def symbolic_lambda(self):
    if self.lambda_expression is None:
      self.lambda_expression = sympy.lambdify([x,y,z], self.compute_symbolic())
    return self.lambda_expression

  def compute_symbolic(self):
    subs = {}

    if type(self.expression) == sympy.Subs:
      gx, gy, gz = self.inputs["g"].compute_symbolic()
      subs["gx"] = gx
      subs["gy"] = gy
      subs["gz"] = gz

    for name, value in self.inputs.items():
      if isinstance(value, Operad):
        subs[name] = value.compute_symbolic()
      else:
        subs[name] = value

    return substitute(self.expression, subs)

  def compute_tensorflow(self, X, Y, Z, **kwargs):
    input_tensors = OrderedDict()

    if type(self.expression) == sympy.Subs:
      gx, gy, gz = self.inputs["g"](X,Y,Z, **kwargs)
      input_tensors["gx"] = gx
      input_tensors["gy"] = gy
      input_tensors["gz"] = gz

    for name, value in self.inputs.items():
      if isinstance(value, Operad):
        input_tensors[name] = value(X,Y,Z, **kwargs)
      else:
        input_tensors[name] = value

    variable_list = [x,y,z] + input_tensors.keys()
    return sympy.lambdify(variable_list, self.expression, "tensorflow")(X,Y,Z,*input_tensors.values())

  def wrap_output(self, func):
    def wrapped(*args, **kwargs):
      return func(self(*args, **kwargs))
    return self.__class__(wrapped, inputs=[self])

  def __lshift__(self, other):
    raise ValueError("Can not compose types: %s and %s" % (type(self), type(other)))

  def __repr__(self):
    return "<Core.Operad."+type(self).__name__+": "+str(self.namespace)+"> {"+str(self.expression)+"}"

def substitute(expr, subs):
  if isinstance(expr, tuple):
    return sympy.Tuple(*[substitute(e, subs) for e in expr])
  substituted = expr.subs(subs)
  if type(substituted) == sympy.Subs:
    return substituted.doit()
  return substituted
