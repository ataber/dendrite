import unittest
from sympy.abc import x,y,z
from dendrite.geometry.primitives.quadrics import sphere

class SphereTest(unittest.TestCase):
  def test_DistanceField(self):
    s = sphere(1)
    self.assertTrue(s(0,0,0) == 1)
    self.assertTrue(s(0,0,1) == 0)
    self.assertTrue(s(0,1,0) == 0)
    self.assertTrue(s(1,0,0) == 0)
    self.assertTrue(s(2,0,0) == -1)
    self.assertTrue(s(5,0,0) == -4)
    self.assertTrue(s(5,0,0) == -4)

  def test_FunctionalArgs(self):
    s = sphere(1-z+x*y)
    self.assertTrue(s(0,0,0) == 1)
