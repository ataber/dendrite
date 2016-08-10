import unittest
import numpy as np
from test.utils import *
from dendrite.transformations.affine import translate, rotate
from dendrite.geometry.primitives.quadrics import *

class BooleanTest(TestCase):
  def test_union(self):
    s1 = sphere(1)
    s2 = sphere(1) << translate([0,0,2])
    u = s1 | s2
    self.assertEqual(u(0,0,0), s1(0,0,0))
    self.assertEqual(u(0,0,1), 0)
    self.assertEqual(u(0,0,1), s2(0,0,1))
    self.assertEqual(u(0,0,2), s2(0,0,2))
    self.assertEqual(u(0,0,3), s2(0,0,3))

    # Ensure Union obeys maximum properties, with wiggle room for FLOP error
    for point in sample_points(5):
      self.assertClose(u(*point), max(s1(*point), s2(*point)))

  def test_intersect(self):
    t1 = torus(1,0.5)
    t2 = torus(1,0.5) << rotate([0,0,1],np.pi/4)
    u = t1 & t2
    self.assertEqual(u(0,0,0), t1(0,0,0))
    self.assertEqual(u(0,0,1), t1(0,0,1))

    # (0.5,0.5,0) is only in t2
    self.assertGreater(t2(0.5,0.5,0), 0)
    self.assertLess(u(0.5,0.5,0), 0)

    # Ensure Intersect obeys minimum properties, with wiggle room for FLOP error
    for point in sample_points(5):
      self.assertClose(u(*point), min(t1(*point), t2(*point)))

  def test_subtract(self):
    s1 = sphere(1)
    s2 = sphere(0.5)
    u = s1 // s2
    self.assertEqual(u(0,0,0), -0.5)
    self.assertEqual(u(0,0.5,0), 0)
    self.assertEqual(u(0,1,0), 0)
    self.assertEqual(u(0,1,1), s1(0,1,1))
