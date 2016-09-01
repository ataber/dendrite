from functools import wraps
from numbers import Number
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
