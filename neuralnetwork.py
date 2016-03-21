import numpy as np


class NeuralNetwork:
    def __init__(self, input_layer_size, output_layer_size, hidden_layer_sizes, initial_thetas=None):
        if type(input_layer_size) != int or type(output_layer_size) != int:
            raise TypeError("Input and output layer sizes must be in int.")

        if input_layer_size <= 0 or output_layer_size <= 0:
            raise ValueError("Input and output layer sizes must be greater than zero.")

        if len(hidden_layer_sizes) < 1:
            raise ValueError("There must be at least one hidden layer.")

        if not all(isinstance(x, int) and x > 0 for x in hidden_layer_sizes):
            raise ValueError("All hidden layer sizes must be positive integer.")

        self.input_layer_size = input_layer_size
        self.output_layer_size = output_layer_size
        self.hidden_layer_sizes = list(hidden_layer_sizes)
        self.hidden_layer_count = len(self.hidden_layer_sizes)
        self.thetas = self._random_initialize_theta() if initial_thetas is None else [np.asarray(t) for t in
                                                                                      initial_thetas]

    @classmethod
    def init(cls, input_layer_size, output_layer_size, hidden_layer_sizes):
        return cls(input_layer_size=input_layer_size, output_layer_size=output_layer_size,
                   hidden_layer_sizes=hidden_layer_sizes, initial_thetas=None)

    @classmethod
    def init_with_theta(cls, thetas):
        if len(thetas) < 3:
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

        return cls(input_layer_size=ils, output_layer_size=ols, hidden_layer_sizes=hls, initial_thetas=thetas)

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
