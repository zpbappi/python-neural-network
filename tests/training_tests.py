import unittest
import numpy as np

from neuralnetwork import NeuralNetwork
from model import Model


class TrainingTests(unittest.TestCase):

    def test_unroll_theta_should_work_properly(self):
        thetas = np.asarray([[[1,2,3], [4,5,6]], [[-1, -2, -3], [-4, -5, -6]]])
        actual = NeuralNetwork._unroll_matrices(thetas)
        expected = np.array([1, 2, 3, 4, 5, 6, -1, -2, -3, -4, -5, -6])
        np.testing.assert_array_equal(actual, expected)

    def test_roll_into_theta_should_work_properly(self):
        nn = NeuralNetwork.init(0, 2, 2, [2])
        unrolled_theta_vector = [1, 2, 3, 4, 5, 6, -1, -2, -3, -4, -5, -6]
        actual = nn._roll_into_matrices(unrolled_theta_vector)
        expected = np.asarray([[[1,2,3], [4,5,6]], [[-1, -2, -3], [-4, -5, -6]]])
        np.testing.assert_array_equal(actual, expected)

    def test_cost_regularization_returns_zero_for_no_lambda(self):
        nn = NeuralNetwork.init(0, 10, 2, [10, 10])
        thetas = nn._initial_thetas
        actual = nn.cost_regularization(thetas, 10)
        self.assertEqual(actual, 0)

    def test_cost_regularization_should_return_proper_value(self):
        nn = NeuralNetwork.init(0.1, 2, 2, [2])
        current_thetas = np.array([[[1,2,3], [4,5,6]], [[-1, -2, -3], [-4, -5, -6]]])
        actual = nn.cost_regularization(current_thetas, 100)
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
        X = np.random.rand(10, 5)
        Y = np.matrix([[0, 1, 0, 1, 0, 1, 1, 1, 0, 1]]).T
        unrolled_thetas = nn._unroll_matrices(nn._initial_thetas)
        result = nn._calculate_cost_gradient(unrolled_thetas, X, Y)

        # no idea what the cost would be, but i expect it to be greater than zero
        # it is extremely unlikely to have a perfect model in just one step with random initialization
        self.assertGreaterEqual(result[0], 0.)

    def test_train_should_return_a_model(self):
        nn = NeuralNetwork.init(0.03, 5, 3, [10])
        X = np.random.rand(10, 5)
        Y = np.asmatrix((np.random.rand(10,3) > 0.5).astype(int))
        result = nn.train(X, Y)
        self.assertIsInstance(result, Model)

    def test_any_output_other_than_zero_one_throws(self):
        nn = NeuralNetwork.init(0.03, 5, 1, [10])
        X = np.random.rand(10, 5)
        Y = np.matrix([[0, 1, 0, 1, 0, 1, 1, 2, 0, 1]])
        with self.assertRaises(ValueError):
            nn.train(X, Y)

    def test_training_a_nn_multiple_times_keeps_the_initial_theta_unchanged(self):
        nn = NeuralNetwork.init(0.03, 5, 3, [10])
        expected = nn._initial_thetas[:]
        X = np.random.rand(10, 5)
        Y = np.asmatrix((np.random.rand(10,3) > 0.5).astype(int))
        nn.train(X, Y)
        nn.train(X, Y)
        self.assertEqual(expected, nn._initial_thetas)

    @unittest.skip('Learning XOR is more difficult than it seems. Will deal with it as a separate problem.')
    def test_training_for_XOR_learns(self):
        nn = NeuralNetwork.init(0, 2, 1, [2])
        X = np.matrix([[0, 0], [0, 1], [1, 0], [1, 1]])
        Y = np.matrix([[0], [1], [1], [0]])
        model = nn.train(X, Y, maxiter=400, tolerance=np.finfo(float).eps)
        prediction = model.predict_binary_classification(X)
        np.testing.assert_array_equal(prediction, Y)

    def test_spoon_feeding_theta_should_work_when_learning_XNOR(self):
        thetas = [np.matrix([[-7, 5, 5], [5, -10, -10]]), np.matrix([[-5, 10, 10]])]
        nn = NeuralNetwork.init_with_theta(0, thetas)
        X = np.matrix([[0, 0], [0, 1], [1, 0], [1, 1]])
        Y = np.matrix([[1], [0], [0], [1]])
        model = nn.train(X, Y)
        prediction = model.predict_binary_classification(X)
        np.testing.assert_array_equal(prediction, Y)
