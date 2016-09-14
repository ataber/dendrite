import sympy
from sympy.abc import t
import tensorflow as tf

def bisection_method(to_minimize, input_tensors, coordinate_shape, bounds=(0,1)):
  condition = lambda i, a, b: i < 30
  def body(i, left, right):
    expression_to_solve = sympy.fraction(sympy.diff(to_minimize, t))[0]
    lambda_to_solve = sympy.lambdify(
      list(input_tensors.keys())+[t],
      expression_to_solve,
      "tensorflow"
    )
    left_residual = lambda_to_solve(*input_tensors.values(), left)
    right_residual = lambda_to_solve(*input_tensors.values(), right)
    mean = (left + right) / 2
    mean_residual = lambda_to_solve(*input_tensors.values(), mean)
    differing_signs = tf.not_equal(tf.sign(left_residual), tf.sign(right_residual))
    same_sign_as_mean = lambda side_residual: tf.equal(tf.sign(side_residual), tf.sign(mean_residual))
    select = lambda side, res: tf.select(tf.logical_and(differing_signs, same_sign_as_mean(res)), mean, side)
    left_conditional = select(left, left_residual)
    right_conditional = select(right, right_residual)
    return (i+1, left_conditional, right_conditional)

  left_bound = tf.constant(bounds[0], shape=coordinate_shape, dtype=tf.float32)
  right_bound = tf.constant(bounds[1], shape=coordinate_shape, dtype=tf.float32)
  _, left, right = tf.while_loop(
    condition,
    body,
    [tf.constant(0), left_bound, right_bound]
  )

  minimize_lambda = sympy.lambdify(
    list(input_tensors.keys())+[t],
    to_minimize,
    "tensorflow"
  )

  left_value = minimize_lambda(*input_tensors.values(), left)
  right_value = minimize_lambda(*input_tensors.values(), right)
  solution = tf.select(tf.less_equal(left_value, right_value), left, right)
  return solution

def newtons_method(to_minimize, input_tensors, coordinate_shape):
  variable_list = list(input_tensors.keys()) + [t]
  to_solve = sympy.fraction(sympy.diff(to_minimize, t))[0]
  lambda_to_solve = sympy.lambdify(variable_list, to_solve, "tensorflow")

  def condition(i, _t):
    residual = lambda_to_solve(*input_tensors.values(), _t)
    return tf.logical_or(i < 200, tf.reduce_mean(abs(residual)) < 1)
  def body(i, _t):
    residual = lambda_to_solve(*input_tensors.values(), _t)
    gradient = tf.gradients(residual, [_t])[0]
    return (i+1, _t - (residual / gradient))

  time = tf.constant(0.5, shape=coordinate_shape, dtype=tf.float32)
  _, solution = tf.while_loop(
    condition,
    body,
    [tf.constant(0), time]
  )
  return solution
