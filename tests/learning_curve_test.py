import numpy as np
import unittest

from learningcurve import LearningCurve

class LearningCurveTest(unittest.TestCase):
    def test_all_x_and_y_parameters_are_mandatory(self):
        with self.assertRaises(ValueError):
            LearningCurve(None, None, None, None)
        with self.assertRaises(ValueError):
            LearningCurve(np.random.rand(10, 5), np.random.rand(10, 1), None, None)

        successful_lc = LearningCurve(np.random.rand(10, 5), np.random.rand(10, 1), np.random.rand(2, 5), np.random.rand(2, 1))

    def test_train_and_cv_data_should_be_in_proper_shape(self):
        with self.assertRaises(ValueError):
            LearningCurve(np.random.rand(10, 5), np.random.rand(9, 1), np.random.rand(2, 5), np.random.rand(2, 1))
        with self.assertRaises(ValueError):
            LearningCurve(np.random.rand(10, 5), np.random.rand(10, 1), np.random.rand(2, 6), np.random.rand(2, 1))
        with self.assertRaises(ValueError):
            LearningCurve(np.random.rand(10, 5), np.random.rand(10, 1), np.random.rand(2, 5), np.random.rand(3, 1))
        with self.assertRaises(ValueError):
            LearningCurve(np.random.rand(10, 5), np.random.rand(10, 1), np.random.rand(2, 5), np.random.rand(2, 2))

        successful_lc = LearningCurve(np.random.rand(10, 5), np.random.rand(10, 1), np.random.rand(2, 5), np.random.rand(2, 1))