import numpy as np
import tensorflow as tf
import sympy

def cos(t):
  if isinstance(t, tf.Tensor):
    return tf.cos(t)
  elif isinstance(t, sympy.Expr):
    return sympy.cos(t)
  else:
    return np.cos(t)

def sin(t):
  if isinstance(t, tf.Tensor):
    return tf.sin(t)
  elif isinstance(t, sympy.Expr):
    return sympy.sin(t)
  else:
    return np.sin(t)

def arcsin(t):
  if isinstance(t, tf.Tensor):
    return tf.asin(t)
  elif isinstance(t, sympy.Expr):
    return sympy.asin(t)
  else:
    return np.arcsin(t)

def arccos(t):
  if isinstance(t, tf.Tensor):
    return tf.acos(t)
  elif isinstance(t, sympy.Expr):
    return sympy.acos(t)
  else:
    return np.arccos(t)

def arctan(t):
  if isinstance(t, tf.Tensor):
    return tf.atan(t)
  elif isinstance(t, sympy.Expr):
    return sympy.atan(t)
  else:
    return np.arctan(t)

def arccot(t):
  return arctan(1/t)
