import unittest
import numpy as np

from neuralnetwork import NeuralNetwork


class TrainingTests(unittest.TestCase):

    def test_unroll_theta_should_work_properly(self):
        thetas = np.asarray([[[1,2,3], [4,5,6]], [[-1, -2, -3], [-4, -5, -6]]])
        actual = NeuralNetwork._unroll_matrices(thetas)
        expected = [1, 2, 3, 4, 5, 6, -1, -2, -3, -4, -5, -6]
        self.assertEqual(actual, expected)

    def test_roll_into_theta_should_work_properly(self):
        nn = NeuralNetwork.init(0, 2, 2, [2])
        unrolled_theta_vector = [1, 2, 3, 4, 5, 6, -1, -2, -3, -4, -5, -6]
        actual = nn._roll_into_matrices(unrolled_theta_vector)
        expected = np.asarray([[[1,2,3], [4,5,6]], [[-1, -2, -3], [-4, -5, -6]]])
        np.testing.assert_array_equal(actual, expected)
