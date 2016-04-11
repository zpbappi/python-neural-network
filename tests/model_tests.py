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

    def test_thetas_should_have_proper_dimensions(self):
        with self.assertRaises(ValueError):
            Model([np.random.rand(5,4), np.random.rand(2, 5)])
        successful_model = Model([np.random.rand(10, 5), np.random.rand(10, 11), np.random.rand(5, 11), np.random.rand(3, 6)])

    def test_model_should_take_a_single_input_row(self):
        model = Model([np.random.rand(10, 5), np.random.rand(10, 11), np.random.rand(5, 11), np.random.rand(3, 6)])
        model.evaluate([x for x in range(4)])

    def test_model_should_also_take_multiple_rows(self):
        model = Model([np.random.rand(10, 5), np.random.rand(10, 11), np.random.rand(5, 11), np.random.rand(3, 6)])
        model.evaluate([[x for x in range(4)], [x**2 for x in range(4)]])

    def test_evaluation_should_throw_for_invalid_input_dimensions(self):
        model = Model([np.random.rand(10, 5), np.random.rand(10, 11), np.random.rand(5, 11), np.random.rand(3, 6)])
        with self.assertRaises(ValueError):
            model.evaluate([x for x in range(2)])

    def test_evaluate_returns_correct_output_shape(self):
        model = Model([np.random.rand(10, 5), np.random.rand(10, 11), np.random.rand(5, 11), np.random.rand(3, 6)])
        result = model.evaluate([[r*5+c for c in range(4)] for r in range(100)])
        actual = result.shape
        expected = (100, 3)
        self.assertEqual(actual, expected)

    def test_evaulate_returns_correct_shapes_for_trivial_cases(self):
        model = Model([np.random.rand(1,2), np.random.rand(1,2)])
        result = model.evaluate(np.random.rand(1,1))
        self.assertEqual(result.shape, (1,1))

    def test_binary_classification_throws_on_more_than_one_output_feature(self):
        model = Model([np.random.rand(4, 6), np.random.rand(3, 5)])
        with self.assertRaises(ValueError):
            model.predict_binary_classification(np.random.rand(10, 5))

    def test_binary_classsification_should_out_row_vector_of_0_1_only(self):
        model = Model([np.random.rand(4, 6), np.random.rand(1, 5)])
        prediction = model.predict_binary_classification(np.random.rand(10, 5))
        self.assertEqual(prediction.shape, (10, 1))
        zeros = len(np.argwhere(prediction == 0))
        ones = len(np.argwhere(prediction == 1))
        self.assertEqual(zeros + ones, 10)

    def test_multiclass_classification_throws_on_one_output_feature(self):
        model = Model([np.random.rand(4, 6), np.random.rand(1, 5)])
        with self.assertRaises(ValueError):
            model.predict_multiclass_classification(np.random.rand(10, 5))

    def test_multiclass_classification_throws_on_two_output_feature(self):
        model = Model([np.random.rand(4, 6), np.random.rand(2, 5)])
        with self.assertRaises(ValueError):
            model.predict_multiclass_classification(np.random.rand(10, 5))

    def test_multiclass_classification_returns_proper_output_shape_and_values(self):
        model = Model([np.random.rand(4, 6), np.random.rand(3, 5)])
        prediction = model.predict_multiclass_classification(np.random.rand(10, 5))
        self.assertEqual(prediction.shape, (10, 1))
        self.assertTrue(all(prediction >= 0))
        self.assertTrue(all(prediction <= 2))