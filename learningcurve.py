import numpy as np
from neuralnetwork import NeuralNetwork

class LearningCurve:
    def __init__(self, X_train, Y_train, X_cv, Y_cv):
        if X_train is None or Y_train is None or X_cv is None or Y_cv is None:
            raise(ValueError("Both training and cross validation data are mandatory for learning curve."))

        X_train = np.asmatrix(X_train)
        Y_train = np.asmatrix(Y_train)
        X_cv = np.asmatrix(X_cv)
        Y_cv = np.asmatrix(Y_cv)

        if X_train.shape[0] != Y_train.shape[0]:
            raise(ValueError("Training input and output data set should have same number of rows."))
        if X_cv.shape[0] != Y_cv.shape[0]:
            raise(ValueError("Cross validation input and output data set should have same number of rows."))
        if X_train.shape[1] != X_cv.shape[1]:
            raise(ValueError("Both training and cross validation input data set should have same number of features."))
        if Y_train.shape[1] != Y_cv.shape[1]:
            raise(ValueError("Both training and cross validation output data set should have same number of features."))
