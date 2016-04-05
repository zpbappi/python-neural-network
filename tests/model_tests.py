import unittest
import numpy as np

from model import Model

class ModelTests(unittest.TestCase):
    def test_thetas_should_be_array_of_at_least_2_elements(self):
        with self.assertRaises(ValueError):
            Model(None)

        with self.assertRaises(ValueError):
            Model([])

        with self.assertRaises(ValueError):
            Model([np.random.rand(5,4)])

        successful_model = Model([np.random.rand(5,4), np.random.rand(2, 6)])
