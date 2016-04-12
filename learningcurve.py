import numpy as np
from neuralnetwork import NeuralNetwork
from math_helper import MathHelper

class LearningCurve:
    def __init__(self, lambda_val, hidden_layer_sizes, X_train, Y_train, X_cv, Y_cv):
        if lambda_val is None:
            raise(ValueError("Lambda value is mandatory."))
        if hidden_layer_sizes is None or len(hidden_layer_sizes) == 0:
            raise(ValueError("Must provide a valid hidden layer size."))
        if X_train is None or Y_train is None or X_cv is None or Y_cv is None:
            raise(ValueError("Both training and cross validation data are mandatory for learning curve."))

        self.X_train = np.asmatrix(X_train)
        self.Y_train = np.asmatrix(Y_train)
        self.X_cv = np.asmatrix(X_cv)
        self.Y_cv = np.asmatrix(Y_cv)
        self._lambda = lambda_val
        self._hidden_layer_sizes = hidden_layer_sizes
        self._helper = MathHelper()


        if self.X_train.shape[0] != self.Y_train.shape[0]:
            raise(ValueError("Training input and output data set should have same number of rows."))
        if self.X_cv.shape[0] != self.Y_cv.shape[0]:
            raise(ValueError("Cross validation input and output data set should have same number of rows."))
        if self.X_train.shape[1] != self.X_cv.shape[1]:
            raise(ValueError("Both training and cross validation input data set should have same number of features."))
        if self.Y_train.shape[1] != self.Y_cv.shape[1]:
            raise(ValueError("Both training and cross validation output data set should have same number of features."))

    def _cost(self, Y, hypothesis, train_data_size, cost_regularization):
        eps = np.finfo(float).eps
        J = sum(-np.log(hypothesis[Y == 1] + eps))
        J += sum(-np.log((1 + eps) - hypothesis[Y == 0]))
        J /= train_data_size
        J += cost_regularization
        return J


    def generate(self, indices = None):
        m, n = self.X_train.shape
        k = self.Y_train.shape[1]
        m_cv = self.X_cv.shape[0]

        nn = NeuralNetwork.init(self._lambda, n, k, self._hidden_layer_sizes)

        indices = range(1, m+1) if indices is None else indices

        for i in indices:
            x_sub = self.X_train[:i,:]
            y_sub = self.Y_train[:i,:]
            model = nn.train(x_sub, y_sub)

            cost_reg_train = nn.cost_regularization(model.thetas, i)
            h_train = model.evaluate(x_sub)
            error_train = self._cost(y_sub, h_train, i, cost_reg_train)

            cost_reg_cv = nn.cost_regularization(model.thetas, m_cv)
            h_cv = model.evaluate(self.X_cv)
            error_cv = self._cost(self.Y_cv, h_cv, m_cv, cost_reg_cv)

            yield (i, error_train, error_cv)
