from dendrite.mathematics.elementary import Min, Max
from dendrite.core.functional import Functional as F
from dendrite.transformations.periodic import saw_wave
from dendrite.decorators.type_coercion import coerce_types

@coerce_types
def replicate(obj: F, input_dimensions=[1,1,1], output_dimensions=[1,1,1], D: str ="xyz") -> F:
  # Multiply transformed output by maximum of gradient of transformation to preserve sub-D-field property
  scaling_factor = Min(*output_dimensions) / Max(*input_dimensions)
  return (obj << saw_wave(input_dimensions, output_dimensions, D)) * scaling_factor
