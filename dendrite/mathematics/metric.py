import numpy as np
import tensorflow as tf

def dot(a, b):
  if isinstance(a, tf.Tensor):
    if isinstance(b, np.ndarray):
      b = tf.constant(b, dtype=tf.float32)
    with tf.name_scope("Dot"):
      return tf.reduce_sum(a * b, reduction_indices=0)
  else:
    return np.dot(a, b)

def norm(vector):
  if isinstance(vector, tf.Tensor):
    with tf.name_scope("Norm"):
      return tf.sqrt(tf.reduce_sum(tf.square(vector), reduction_indices=0))
  else:
    return np.linalg.norm(vector)
