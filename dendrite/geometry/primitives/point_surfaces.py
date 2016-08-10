from scipy import linalg
from dendrite.core.functional import Functional as F
from dendrite.core.expression import Expression as E
from dendrite.mathematics.elementary import *

@E
def Blobby(*points, threshold=1, N=2) -> F:
  potentials = [
    exp(-sum((np.array([x,y,z]) - np.array(point))**N)**(1/N)) for point in points
  ]
  return sum(potentials, -threshold)

#https://smartech.gatech.edu/bitstream/handle/1853/3382/99-15.pdf
@E
def variational_surface(*zero_points, interior_constraints=[], exterior_constraints=[]) -> F:
  if len(interior_constraints) == 0 and len(exterior_constraints) == 0:
    # choose centroid
    interior_constraints.append(np.mean(zero_points, 0))
  radial_basis_function = lambda x: x**3
  norm = lambda x,y,z: sqrt(x**2 + y**2 + z**2)
  points = [np.array(point) for point in list(zero_points) + interior_constraints + exterior_constraints]
  k = len(points)
  matrix = np.zeros((k, k))
  for i in range(k):
    for j in range(k):
      matrix[i, j] = radial_basis_function(norm(*(points[i] - points[j])))

  b = np.zeros(k)
  for i in range(len(interior_constraints)):
    b[len(zero_points)+i] = 1
  for i in range(len(exterior_constraints)):
    b[len(zero_points)+len(interior_constraints)+i] = -1

  lu_factorization = linalg.lu_factor(matrix)
  solution = linalg.lu_solve(lu_factorization, b)

  return sum([
    weight * radial_basis_function(norm(*(point - [x,y,z]))) for weight, point in zip(solution, points)
  ])
