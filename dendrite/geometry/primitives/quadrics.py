from sympy.abc import x,y,z
from dendrite.core.functional import Functional as F
from dendrite.core.expression import Expression as E
from dendrite.mathematics.elementary import sqrt, Min

@E
def sphere(radius: float) -> F:
  return radius - sqrt(x**2 + y**2 + z**2)

@E
def torus(major_radius: float, minor_radius: float) -> F:
  return minor_radius - sqrt((sqrt(y**2 + z**2) - major_radius)**2 + x**2)

@E
def ellipsoid(x_radius: float, y_radius: float, z_radius: float) -> F:
  return (1 - ((x / x_radius)**2 + (y / y_radius)**2 + (z / z_radius)**2)) * Min(x_radius, y_radius, z_radius)

@E
def superquadric(r: float, s: float, t: float) -> F:
  return 1 - abs(x)**r - abs(y)**s - abs(z)**t

@E
def sphere_squared(radius: float) -> F:
  # This returns the squared distance field. Same isosurface as sphere. Required for channel surfaces.
  return radius**2 - (x**2 + y**2 + z**2)

@E
def cylinder(radius: float) -> F:
  return sphere(radius)(x,y,0)
