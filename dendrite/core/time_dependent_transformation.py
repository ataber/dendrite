from dendrite.core.transformation import Transformation
from dendrite.core.operad import Operad

class TimeDependentTransformation(Transformation):
  def __init__(self, expressions, namespace=None):
    # expressions is a tuple of (time-dependent tuple of equations, equation to solve numerically for time)
    expression, to_solve = expressions
    Operad.__init__(self, expression, namespace=namespace, to_solve=to_solve)
