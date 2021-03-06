import numpy as np
import tensorflow as tf
import sympy

def factorial(n):
  if n==0:
    val=1
  else:
    val=np.prod(np.arange(1,float(n+1)))
  return val

def choose(n,k):
  return factorial(n) / (factorial(k) * factorial(n-k))

def Max(*args):
  return sympy.Max(*args)

def Min(*args):
  return sympy.Min(*args)

def bound(t, a, b):
  return Min(Max(t, a), b)

def floor(t):
  if isinstance(t, sympy.Expr):
    return sympy.floor(t)
  else:
    return np.floor(t)

def reduce_sum(t):
  if isinstance(t, tf.Tensor):
    return tf.reduce_sum(t, reduction_indices=0)
  else:
    return np.sum(t, axis=0)

def sqrt(t):
  if isinstance(t, sympy.Expr):
    return sympy.sqrt(t)
  else:
    return np.sqrt(t)

def exp(t):
  if isinstance(t, sympy.Expr):
    return sympy.exp(t)
  else:
    return np.exp(t)

def log(t):
  if isinstance(t, sympy.Expr):
    return sympy.log(t)
  else:
    return np.log(t)
