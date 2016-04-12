import numpy as np
import unittest

from learningcurve import LearningCurve

class LearningCurveTest(unittest.TestCase):
    def test_all_x_and_y_parameters_are_mandatory(self):
        with self.assertRaises(ValueError):
            LearningCurve(0, [2], None, None, None, None)
        with self.assertRaises(ValueError):
            LearningCurve(0, [10], np.random.rand(10, 5), np.random.rand(10, 1), None, None)

        successful_lc = LearningCurve(0, [5], np.random.rand(10, 5), np.random.rand(10, 1), np.random.rand(2, 5), np.random.rand(2, 1))

    def test_lambda_and_hidden_layer_sizes_are_mandatory(self):
        with self.assertRaises(ValueError):
            LearningCurve(None, [5], np.random.rand(10, 5), np.random.rand(10, 1), np.random.rand(2, 5), np.random.rand(2, 1))
        with self.assertRaises(ValueError):
            LearningCurve(0, [], np.random.rand(10, 5), np.random.rand(10, 1), np.random.rand(2, 5), np.random.rand(2, 1))
        with self.assertRaises(ValueError):
            LearningCurve(0, None, np.random.rand(10, 5), np.random.rand(10, 1), np.random.rand(2, 5), np.random.rand(2, 1))

        successful_lc = LearningCurve(0, [5], np.random.rand(10, 5), np.random.rand(10, 1), np.random.rand(2, 5), np.random.rand(2, 1))

    def test_train_and_cv_data_should_be_in_proper_shape(self):
        with self.assertRaises(ValueError):
            LearningCurve(0, [5], np.random.rand(10, 5), np.random.rand(9, 1), np.random.rand(2, 5), np.random.rand(2, 1))
        with self.assertRaises(ValueError):
            LearningCurve(0, [5], np.random.rand(10, 5), np.random.rand(10, 1), np.random.rand(2, 6), np.random.rand(2, 1))
        with self.assertRaises(ValueError):
            LearningCurve(0, [5], np.random.rand(10, 5), np.random.rand(10, 1), np.random.rand(2, 5), np.random.rand(3, 1))
        with self.assertRaises(ValueError):
            LearningCurve(0, [5], np.random.rand(10, 5), np.random.rand(10, 1), np.random.rand(2, 5), np.random.rand(2, 2))

        successful_lc = LearningCurve(0, [5], np.random.rand(10, 5), np.random.rand(10, 1), np.random.rand(2, 5), np.random.rand(2, 1))

    def test_learning_curve_generate_should_return_generator(self):
        import types
        lc = LearningCurve(0, [3], np.random.rand(10, 5), (np.random.rand(10, 3) > 0.5).astype(int), np.random.rand(2, 5), (np.random.rand(2, 3) > 0.5).astype(int))
        data_points = lc.generate()
        self.assertIsInstance(data_points, types.GeneratorType)

    def test_learning_curve_generates_proper_touples(self):
        lc = LearningCurve(0, [3], np.random.rand(10, 5), (np.random.rand(10, 3) > 0.5).astype(int), np.random.rand(2, 5), (np.random.rand(2, 3) > 0.5).astype(int))
        data_points = lc.generate()
        first_point = data_points.__next__()
        self.assertEqual(len(first_point), 3)
        self.assertIsInstance(first_point[0], int)
        self.assertEqual(first_point[0], 1)

    def test_learning_curve_generator_should_work_with_custom_indices(self):
        lc = LearningCurve(0, [3], np.random.rand(10, 5), (np.random.rand(10, 3) > 0.5).astype(int), np.random.rand(2, 5), (np.random.rand(2, 3) > 0.5).astype(int))
        custom_indices = [1, 3, 4];
        data_points = lc.generate(custom_indices)
        data_sizes = [x for (x, error_train, error_cv) in data_points]
        self.assertEqual(custom_indices, data_sizes)
