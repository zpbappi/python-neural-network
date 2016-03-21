import unittest
from unittest import mock
from neuralnetwork import NeuralNetwork


class NNConstructionTest(unittest.TestCase):
    def test_cannot_construct_without_any_param(self):
        with self.assertRaises(TypeError):
            NeuralNetwork()

    def test_input_and_output_sizes_must_be_int(self):
        with self.assertRaises(TypeError):
            NeuralNetwork("A", "B", [42])
        with self.assertRaises(TypeError):
            NeuralNetwork(1, "B", [42])
        with self.assertRaises(TypeError):
            NeuralNetwork("A", 1, [42])

        nn = NeuralNetwork(1, 1, [42])

    def test_input_and_output_size_must_be_positive(self):
        with self.assertRaises(ValueError):
            NeuralNetwork(0, 1, [42])
        with self.assertRaises(ValueError):
            NeuralNetwork(1, 0, [42])

    def test_there_should_be_at_least_one_hidden_layer(self):
        with self.assertRaises(ValueError):
            NeuralNetwork(10, 10, [])

    def test_all_hidden_layer_sizes_must_be_positive_int(self):
        with self.assertRaises(ValueError):
            NeuralNetwork(10, 10, [-1])
            NeuralNetwork(10, 10, [1.23])

        NeuralNetwork(10, 10, [1, 2, 3])
        NeuralNetwork(10, 10, (1, 42))

    def test_should_persist_all_the_ctor_args(self):
        nn = NeuralNetwork(100, 42, (1, 2, 3, 4))
        self.assertEqual(nn.input_layer_size, 100)
        self.assertEqual(nn.output_layer_size, 42)
        self.assertEqual(nn.hidden_layer_count, 4)
        self.assertEqual(nn.hidden_layer_sizes, [1, 2, 3, 4])

    def test_should_have_valid_size_for_thetas(self):
        nn = NeuralNetwork(100, 10, (25, 50))

        self.assertEqual(len(nn.thetas), 3)
        self.assertEquals(nn.thetas[0].shape, (25, 101))
        self.assertEquals(nn.thetas[1].shape, (50, 26))
        self.assertEquals(nn.thetas[2].shape, (10, 51))

    def test_factory_class_method_with_sizes_calls_proper_ctor(self):
        mock_init = mock.Mock(return_value=None)
        with mock.patch("neuralnetwork.NeuralNetwork.__init__", new=mock_init):
            instance = NeuralNetwork.init(100, 42, [1, 2, 3])
            mock_init.assert_called_once_with(input_layer_size=100, output_layer_size=42, hidden_layer_sizes=[1, 2, 3],
                                              initial_thetas=None)

    def test_factory_class_method_with_theta_calls_proper_ctor(self):
        thetas = []
        thetas.append([[0.1, 0.2, 0.03],
                       [0.01, 0.02, 0.5],
                       [1, 2, 3]])
        thetas.append([[1, 2, 3, 4],
                       [1, 2, 3, 4],
                       [1, 2, 3, 4],
                       [1, 2, 3, 4]])
        thetas.append([[1, 2, 3, 4, 5]])

        mock_init = mock.Mock(return_value=None)

        with mock.patch("neuralnetwork.NeuralNetwork.__init__", new=mock_init):
            instance = NeuralNetwork.init_with_theta(thetas)
            mock_init.assert_called_once_with(input_layer_size=2, output_layer_size=1, hidden_layer_sizes=[3, 4],
                                              initial_thetas=thetas)

    def test_math_helper_is_created(self):
        nn = NeuralNetwork(100, 42, (1, 2, 3, 4))
        self.assertIsNotNone(nn.helper)