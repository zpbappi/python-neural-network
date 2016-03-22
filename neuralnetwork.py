import numpy as np
import functools

from math_helper import MathHelper


class NeuralNetwork:
    def __init__(self, lambda_val, input_layer_size, output_layer_size, hidden_layer_sizes, initial_thetas=None):
        if lambda_val is None or (type(lambda_val) != float and type(lambda_val) != int):
            raise TypeError("Lambda must be a numeric type.")

        if lambda_val < 0.:
            raise ValueError("Lambda must be a non-negative number.")

        if type(input_layer_size) != int or type(output_layer_size) != int:
            raise TypeError("Input and output layer sizes must be in int.")

        if input_layer_size <= 0 or output_layer_size <= 0:
            raise ValueError("Input and output layer sizes must be greater than zero.")

        if len(hidden_layer_sizes) < 1:
            raise ValueError("There must be at least one hidden layer.")

        if not all(isinstance(x, int) and x > 0 for x in hidden_layer_sizes):
            raise ValueError("All hidden layer sizes must be positive integer.")

        self.lambda_val = lambda_val
        self.input_layer_size = input_layer_size
        self.output_layer_size = output_layer_size
        self.hidden_layer_sizes = list(hidden_layer_sizes)
        self.hidden_layer_count = len(self.hidden_layer_sizes)
        self.thetas = self._random_initialize_theta() if initial_thetas is None else [np.asarray(t) for t in
                                                                                      initial_thetas]
        self.helper = MathHelper()

    @classmethod
    def init(cls, lambda_val, input_layer_size, output_layer_size, hidden_layer_sizes):
        return cls(lambda_val=lambda_val, input_layer_size=input_layer_size, output_layer_size=output_layer_size,
                   hidden_layer_sizes=hidden_layer_sizes, initial_thetas=None)

    @classmethod
    def init_with_theta(cls, lambda_val, thetas):
        if len(thetas) < 2:
            raise ValueError("There must be at least one hidden layer and hence at least three weight matrices.")

        def _shape(item):
            if hasattr(item, "shape"):
                return item.shape
            r = len(item)
            c = 0
            if r > 0:
                c = len(item[0])
            return r, c

        ils = _shape(thetas[0])[1] - 1
        hls = [_shape(t)[0] for t in thetas]
        ols = hls.pop()

        return cls(lambda_val=lambda_val, input_layer_size=ils, output_layer_size=ols, hidden_layer_sizes=hls,
                   initial_thetas=thetas)

    def _random_initialize_theta(self):
        delta = 0.12

        def random_matrix(row, col):
            return np.random.rand(row, col) * 2 * delta - delta

        first_size_array = [self.input_layer_size]
        [first_size_array.append(x) for x in self.hidden_layer_sizes]
        paired_copy = first_size_array[1:]
        paired_copy.append(self.output_layer_size)
        sizes = zip(paired_copy, first_size_array)

        return [random_matrix(r, c + 1) for (r, c) in sizes]

    @staticmethod
    def _unroll_matrices(matrices):
        def mapper(x):
            return np.ravel(x).tolist()

        def reducer(x, y):
            return x + y

        return functools.reduce(reducer, map(mapper, matrices))

    def _roll_into_matrices(self, unrolled_vector):
        taken, prev_layer_size, current_layer_size = 0, self.input_layer_size, self.hidden_layer_sizes[0]
        matrices = []
        for i in range(self.hidden_layer_count):
            current_layer_size = self.hidden_layer_sizes[i]
            matrices.append(np.reshape(unrolled_vector[taken : taken + current_layer_size*(prev_layer_size+1)], (current_layer_size, prev_layer_size+1)))
            taken += current_layer_size * (prev_layer_size + 1)
            prev_layer_size = current_layer_size

        matrices.append(np.reshape(unrolled_vector[taken:], (self.output_layer_size, prev_layer_size+1)))
        return matrices

    def _cost_regularization(self, current_thetas, train_data_size):
        if self.lambda_val == 0:
            return 0

        def mapper(x):
            matrix = np.zeros(x.shape)
            matrix[:,1:] = x[:,1:]
            return np.multiply(matrix, matrix).sum()

        def reducer(x, y):
            return x + y

        return self.lambda_val * functools.reduce(reducer, map(mapper, current_thetas), 0) / (2. * train_data_size)

    def _theta_regularization(self, theta, train_data_size):
        result = np.zeros(theta.shape)

        if self.lambda_val == 0:
            return result

        result[:, 1:] = (self.lambda_val / train_data_size) * theta[:, 1:]
        return result
