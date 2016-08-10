import unittest
import numpy as np
from sympy import *
from sympy.abc import x,y,z
from test.utils import *
from dendrite.geometry.operations.replication import *
from dendrite.transformations.affine import *
from dendrite.geometry.primitives.quadrics import *

class ReplicationTest(TestCase):
  def test_Replicate(self):
    s1 = torus(0.25,0.1) << rotate([0.3,.25,0.1],0.4)
    s2 = replicate(s1, [1,1,1], [1,1,1])
    self.assertEqual(s2(0,0,0), s1(0,0,0))
    self.assertClose(s2(2,2,2), s1(0,0,0))
    self.assertClose(s2(0.4,0.3,0), s1(0.4,0.3,0))
    self.assertClose(s2(-1,2.4,1), s1(0,0.4,0))
    self.assertClose(s2(-1.4,2.4,1), s1(-0.4,0.4,0))
    self.assertClose(s2(0,1,0), s1(0,0,0))

    for point in sample_points(5, [-5,-5,-5], [5,5,5]):
      mod = np.mod(point + 0.5, 1) - 0.5
      self.assertClose(s2(*point), s1(*mod))

  def test_Replicate_with_functional_args(self):
    s1 = torus(0.4,0.1) | (torus(0.4,0.1) << rotate([1,0,0],np.pi))
    p = 1/(floor(((x)**2 + (y)**2)**0.5)+1)
    s2 = replicate(s1, [1,1,1], [p,p,p])
    self.assertEqual(s2(0,0,0), s1(0,0,0))
    self.assertEqual(s2(1,0,0), s1(0,0,0)/2)
    self.assertEqual(s2(1.5,0,0), s1(0,0,0)/2)
    self.assertEqual(s2(0,1.5,0), s1(0,0,0)/2)
    self.assertClose(s2(1.2,0,0), s1(0.4,0,0)/2)
    self.assertClose(s2(2.1,0,0), s1(0.3,0,0)/3)

  def test_Replicate_with_asymmetric_continuous_args(self):
    s1 = torus(0.4,0.1) | (torus(0.4,0.1) << rotate([1,0,0],np.pi))
    p = 1/((((x)**2 + (y)**2)**0.5)+1)
    s2 = replicate(s1, [1,1,1], [p,1,1])
    self.assertEqual(s2(0,0,0), s1(0,0,0))
    self.assertEqual(s2(1,0,0), s1(0,0,0)/2)
    self.assertClose(s2(2,0,0), s1(0,0,0)/3)
    # If we are at y=1 then the distance field should underestimate the true distance by a factor of 2
    self.assertClose(s2(0,1,0), s1(0,0,0)/2)

  def test_Replicate_with_asymmetric_dilation(self):
    s1 = torus(0.4,0.1) | (torus(0.4,0.1) << rotate([1,0,0],np.pi))
    p = (((x)**2 + (y)**2)**0.5)+1
    s2 = replicate(s1, [1,1,1], [p,p,1])
    self.assertEqual(s2(0,0,0), s1(0,0,0))
    # Ensure sub-distance properties
    self.assertLess(abs(s2(1,0,0)), abs(2*s1(0,0,0)))
    self.assertLess(abs(s2(1,1,0)), abs(3*s1(0,0,0)))
    self.assertLess(abs(s2(2,0,0)), abs(3*s1(0,0,0)))

  def test_Replicate_with_different_size_unit_cell(self):
    s1 = torus(0.4,0.1) | (torus(0.4,0.1) << rotate([1,0,0],np.pi))
    s2 = replicate(s1, [2,2,2], [1,1,1])
    self.assertEqual(s2(0,0,0), s1(0,0,0)/2)
    self.assertEqual(s2(0.5,0,0), s1(1,0,0)/2)
    self.assertEqual(s2(0.5,0.5,0), s1(1,1,0)/2)

  def test_Replicate_with_asymmetric_unit_cell(self):
    s1 = torus(0.4,0.1) | (torus(0.4,0.1) << rotate([1,0,0],np.pi))
    s2 = replicate(s1, [2,1,1], [1,1,1])
    self.assertEqual(s2(0,0,0), s1(0,0,0)/2)
    self.assertEqual(s2(0.5,0,0), s1(1,0,0)/2)
    self.assertEqual(s2(0.5,0.5,0), s1(1,0.5,0)/2)
