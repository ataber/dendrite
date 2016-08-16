from dendrite.mathematics.elementary import Min, Max
from dendrite.core.functional import Functional as F
from dendrite.core.expression import Expression as E
from dendrite.transformations.periodic import saw_wave

@E
def replicate(obj: F, input_dimensions: list = [1,1,1], output_dimensions: list = [1,1,1], D: str = "xyz") -> F:
  # Multiply transformed output by maximum of gradient of transformation to preserve sub-D-field property
  scaling_factor = Min(*output_dimensions) / Max(*input_dimensions)
  return (obj << saw_wave(input_dimensions, output_dimensions, D)) * scaling_factor
