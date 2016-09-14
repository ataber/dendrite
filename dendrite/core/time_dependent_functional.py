from dendrite.core.functional import Functional
from dendrite.core.operad import Operad

class TimeDependentFunctional(Functional):
  def __init__(self, expressions, namespace=None):
    # expressions is a tuple of (time-dependent equation, equation to numerically minimize)
    expression, to_minimize = expressions
    Operad.__init__(self, expression, namespace=namespace, to_minimize=to_minimize)
