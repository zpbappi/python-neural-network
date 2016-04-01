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

    def test_cost_regularization_returns_zero_for_no_lambda(self):
        nn = NeuralNetwork.init(0, 10, 2, [10, 10])
        thetas = nn.thetas
        actual = nn._cost_regularization(thetas, 10)
        self.assertEqual(actual, 0)

    def test_cost_regularization_should_return_proper_value(self):
        nn = NeuralNetwork.init(0.1, 2, 2, [2])
        current_thetas = np.array([[[1,2,3], [4,5,6]], [[-1, -2, -3], [-4, -5, -6]]])
        actual = nn._cost_regularization(current_thetas, 100)
        expected = 0.074
        self.assertAlmostEqual(actual, expected)

    def test_theta_regularization_should_return_zero_for_no_lambda(self):
        nn = NeuralNetwork.init(0, 10, 2, [10, 10])
        theta = np.array([[1,2,3], [4,5,6]])
        actual = nn._theta_regularization(theta, 100)
        expected = np.zeros(theta.shape)
        np.testing.assert_array_equal(actual, expected)

    def test_tehta_regularization_should_return_proper_value(self):
        nn = NeuralNetwork.init(0.5, 2, 3, [2])
        theta = np.array([[1,2,3], [4,5,6]])
        actual = nn._theta_regularization(theta, 100)
        expected = np.array([[0, 0.01, 0.015], [0, 0.025, 0.03]])
        np.testing.assert_array_equal(actual, expected)

    def test_gradient_calculation_should_not_throw(self):
        nn = NeuralNetwork.init(0.03, 5, 1, [10])
        X = np.random.rand(10, 6)
        X[:,0] = 1
        Y = np.array([0, 1, 0, 1, 0, 1, 1, 1, 0, 1]);
        unrolled_thetas = nn._unroll_matrices(nn.thetas)
        result = nn._calculate_cost_gradient(unrolled_thetas, X, Y)

        # no idea what the cost would be, but i expect it to be greater than zero
        # it is extremely unlikely to have a perfect model in just one step with a single input & random initialization
        self.assertGreaterEqual(result[0], 0.)
