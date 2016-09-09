from functools import wraps
from numbers import Number
from typing import List
import inspect
import numpy as np
import sympy

def convert_to_operad(cls, obj):
  if isinstance(obj, Number):
    return cls(sympy.sympify(obj), namespace=str(obj))
  elif isinstance(obj, cls):
    return obj
  else:
    return cls(obj, namespace=str(obj))

def coerce_output(func):
  @wraps(func)
  def coerced_output(*args, **kwargs):
    return_annotation = inspect.signature(func).return_annotation
    if return_annotation is not inspect.Parameter.empty:
      result = func(*args, **kwargs)
      if not isinstance(result, return_annotation): result = return_annotation(result)
      if "name" in kwargs:
        result.namespace = kwargs["name"]
      else:
        result.namespace = func.__name__.title()
      return result
    else:
      return func(*args, **kwargs)
  return coerced_output

rational_vector = List[sympy.Rational]
registered_converters = {
  rational_vector: lambda x: [sympy.sympify(str(el), rational=True) for el in x]
}

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
        if param.annotation is not inspect.Parameter.empty:
          if issubclass(param.annotation, List):
            coerced_args[i] = converter(args[i])
          elif isinstance(args[i], param.annotation):
            coerced_args[i] = args[i]
          else:
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
