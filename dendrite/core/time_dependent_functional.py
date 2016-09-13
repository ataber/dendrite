from dendrite.core.functional import Functional
from dendrite.core.operad import Operad

class TimeDependentFunctional(Functional):
  def __init__(self, expressions, namespace=None):
    # expressions is a tuple of (time-dependent equation, equation to solve numerically for time)
    expression, to_solve = expressions
    Operad.__init__(self, expression, namespace=namespace, to_solve=to_solve)
