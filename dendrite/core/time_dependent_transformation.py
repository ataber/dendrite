from dendrite.core.transformation import Transformation
from dendrite.core.operad import Operad

class TimeDependentTransformation(Transformation):
  def __init__(self, expressions, namespace=None):
    # expressions is a tuple of (time-dependent tuple of equations, equation to numerically minimize, bounds on time)
    expression, to_minimize, time_bounds = expressions
    Operad.__init__(self, expression, namespace=namespace, to_minimize=to_minimize, time_bounds=time_bounds)
