from sympy.abc import x,y,z
from dendrite.core.functional import Functional as F
from dendrite.core.expression import Expression as E
from dendrite.mathematics.elementary import sqrt, Min

@E
def sphere(radius) -> F:
  return radius - sqrt(x**2 + y**2 + z**2)

@E
def torus(major_radius, minor_radius) -> F:
  return minor_radius - sqrt((sqrt(y**2 + z**2) - major_radius)**2 + x**2)

@E
def ellipsoid(x_radius, y_radius, z_radius) -> F:
  return (1 - ((x / x_radius)**2 + (y / y_radius)**2 + (z / z_radius)**2)) * Min(x_radius, y_radius, z_radius)

@E
def superquadric(r, s, t) -> F:
  return 1 - abs(x)**r - abs(y)**s - abs(z)**t

@E
def sphere_squared(radius) -> F:
  # This returns the squared distance field. Same isosurface as sphere. Required for channel surfaces.
  return radius**2 - (x**2 + y**2 + z**2)

@E
def cylinder(radius) -> F:
  return sphere(radius)(x,y,0)
