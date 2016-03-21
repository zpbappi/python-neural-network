import unittest
import numpy as np
import scipy as sp

from math_helper import MathHelper


class MathHelperTests(unittest.TestCase):
    def setUp(self):
        self.helper = MathHelper()

    def test_sigmoid(self):
        x = np.asarray([[1, 2, 3], [2, 3, 4]])
        expected = 1. / (1 + sp.exp(-x))
        actual = self.helper.sigmoid(x)
        np.testing.assert_array_equal(actual, expected)

    def test_sigmoid_grad(self):
        x = np.random.rand(5,3)
        sigmoid = 1. / (1 + sp.exp(-x))
        expected = np.multiply(sigmoid, 1 - sigmoid)
        actual = self.helper.sigmoid_grad(x)
        np.testing.assert_array_equal(actual, expected)