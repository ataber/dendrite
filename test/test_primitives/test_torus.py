import unittest
from sympy.abc import x,y,z
from dendrite.geometry.primitives.quadrics import torus

class TorusTest(unittest.TestCase):
  def test_DistanceField(self):
    s = torus(1,0.5)
    self.assertEqual(s(0,0,0), -0.5)
    self.assertEqual(s(0,0,1), 0.5)
    self.assertEqual(s(0,1,0), 0.5)
    self.assertEqual(s(1,0,0), -0.9142135623730951)
    self.assertEqual(s(2,0,0), -1.7360679774997898)
    self.assertEqual(s(0,5,0), -3.5)
    self.assertEqual(s(0,0,5), -3.5)

  def test_FunctionalArgs(self):
    s = torus(1-z+x*y, x+3)
    self.assertEqual(s(0,0,0), 2)
