import unittest
import numpy as np
from random import uniform
from functools import reduce

def sample_point(min_bounds, max_bounds):
  return np.array([uniform(min_bounds[i], max_bounds[i]) for i in range(3)])

def sample_points(num, min_bounds=[-1,-1,-1], max_bounds=[1,1,1]):
  return [sample_point(min_bounds, max_bounds) for _ in range(num)]

class TestCase(unittest.TestCase):
  def assertClose(self, a, b):
    self.assertLessEqual(abs(a - b), 1e-6)

  def assertAllClose(self, *args):
    reduce(self.assertClose, args)
