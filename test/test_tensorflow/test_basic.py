import unittest
import tensorflow as tf
from dendrite.geometry.primitives.quadrics import sphere
from dendrite.transformations.domain import absolute_value

class SphereTest(unittest.TestCase):
  def test_DistanceField(self):
    s = sphere(1)
    zero = tf.constant(0.0)
    one = tf.constant(1.0)
    sess = tf.Session()
    c = lambda tensor: sess.run(tensor)
    self.assertTrue(c(s(zero,zero,zero)) == 1)
    self.assertTrue(c(s(0,zero,one)) == 0)
    self.assertTrue(c(s(zero,one,zero)) == 0)
    self.assertTrue(c(s(one,0,0)) == 0)
    self.assertTrue(c(s(one+one,0,0)) == -1)
    self.assertTrue(c(s(one*5,0,0)) == -4)

  def test_namespace(self):
    s = sphere(1)
    self.assertEqual(s.namespace, "Sphere")
    composed = s << absolute_value()
    self.assertEqual(composed.namespace, "Composition")
