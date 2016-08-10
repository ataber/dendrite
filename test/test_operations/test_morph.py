import unittest
import numpy as np
from test.utils import *
from dendrite.geometry.operations.morph import *
from dendrite.geometry.primitives.quadrics import *

class MorphTest(TestCase):
  def test_scale(self):
    r = 1
    s1 = sphere(r)
    scaled = scale(s1, 2)
    self.assertEqual(scaled(0,0,0), 2*s1(0,0,0))
    self.assertEqual(scaled(0,1,0), 1)
    self.assertEqual(scaled(2,0,0), 0)

    # Ensure Scale preserves d-field properties
    for point in sample_points(5):
      self.assertClose(2*r - np.linalg.norm(point), scaled(*point))

  def test_morph(self):
    t1 = torus(2,1)
    s1 = sphere(1)
    m = lambda t: morph(t1,s1,t)

    for point in sample_points(5):
      self.assertClose(m(1)(*point), t1(*point))
      self.assertClose(m(0)(*point), s1(*point))
      self.assertClose(m(0.5)(*point), (t1(*point) + s1(*point))/2)
