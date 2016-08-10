from functools import wraps, partial
from typing import List
import inspect
import numpy as np
import sympy

registered_converters = {
  # np.array is not a class, just a function for creating ndarrays
  np.ndarray: np.array,
  sympy.Expr: sympy.sympify
}

def wrap_output(cls):
  def decorate(func):
    # Pass Functional output name to namespace TF subgraph
    @wraps(func)
    def namespaced(*args, **kwargs):
      result = func(*args, **kwargs)
      if not isinstance(result, cls): result = cls(result)
      if "name" in kwargs:
        result.namespace = kwargs["name"]
      else:
        result.namespace = func.__name__.title()
      return result
    return namespaced
  return decorate

def coerce_output(func):
  @wraps(func)
  def coerced_output(*args, **kwargs):
    return_annotation = inspect.signature(func).return_annotation
    if return_annotation is not inspect.Parameter.empty:
      wrapped = wrap_output(return_annotation)(func)
      return wrapped(*args, **kwargs)
    else:
      return func(*args, **kwargs)
  return coerced_output

def coerce_custom_types(converter_dict):
  converters = {**converter_dict, **registered_converters}
  return partial(coerce_types, converters=converters)

def coerce_types(func, converters=registered_converters):
  @wraps(func)
  def coerced_types(*args, **kwargs):
    signature = inspect.signature(func)
    coerced_args = [None]*len(args)
    coerced_kwargs = {}
    for i, (name, param) in enumerate(signature.parameters.items()):
      if param.annotation in converters:
        # Use override converter
        converter = converters[param.annotation]
      else:
        # Use class instantiation to convert
        converter = param.annotation

      if i < len(args):
        if not isinstance(args[i], param.annotation) and param.annotation is not inspect.Parameter.empty:
          coerced_args[i] = converter(args[i])
        else:
          coerced_args[i] = args[i]
      else:
        kw_value = kwargs.get(name, None) or signature.parameters[name].default
        if not isinstance(kw_value, param.annotation) and param.annotation is not inspect.Parameter.empty:
          coerced_kwargs[name] = converter(kw_value)
        else:
          coerced_kwargs[name] = kw_value

    if any([arg is inspect.Parameter.empty for arg in coerced_kwargs]):
      # FIXME: if the decorated function is called with wrong number of args, we will assign extra kwargs to empty values
      raise TypeError("Missing arguments in function: " + func.__name__ + "args: " + args + ", kwargs: " + kwargs)

    if signature.return_annotation is not inspect.Parameter.empty:
      wrapped = wrap_output(signature.return_annotation)(func)
      return wrapped(*coerced_args, **coerced_kwargs)
    else:
      return func(*coerced_args, **coerced_kwargs)
  return coerced_types
