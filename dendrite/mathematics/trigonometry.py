import numpy as np
import sympy

def cos(t):
  if isinstance(t, sympy.Basic):
    return sympy.cos(t)
  else:
    return np.cos(t)

def sin(t):
  if isinstance(t, sympy.Basic):
    return sympy.sin(t)
  else:
    return np.sin(t)

def arcsin(t):
  if isinstance(t, sympy.Basic):
    return sympy.asin(t)
  else:
    return np.arcsin(t)

def arccos(t):
  if isinstance(t, sympy.Basic):
    return sympy.acos(t)
  else:
    return np.arccos(t)

def arctan(t):
  if isinstance(t, sympy.Basic):
    return sympy.atan(t)
  else:
    return np.arctan(t)

def arccot(t):
  return arctan(1/t)
