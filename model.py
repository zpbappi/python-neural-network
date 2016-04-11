import numpy as np
from math_helper import MathHelper

class Model:
    def __init__(self, thetas):
        def has_valid_dimensions(matrics):
            collection1 = matrics[:-1]
            collection2 = matrics[1:]
            for (x, y) in zip(collection1, collection2):
                rx, _ = x.shape
                _, ry = y.shape
                if rx + 1 != ry:
                    return False
            return True

        if thetas is None or type(thetas) != list or len(thetas) < 2:
            raise(ValueError("Thetas must be an array of at least 2 matrices."))

        if not has_valid_dimensions(thetas):
            raise(ValueError("Thetas should have dimensions like: (a,b), (b+1,c), (c+1,d)..."))

        self._thetas = thetas
        self._math_helper = MathHelper()

    def evaluate(self, X_in):
        X = np.asmatrix(X_in)
        m, n = X.shape
        theta_0_cols = self._thetas[0].shape[1]
        if n+1 != theta_0_cols:
            raise(ValueError("Input has {0} features and expected theta_0 to have {1} columns, but found {2} columns in theta_0.".format(n, n+1, theta_0_cols)))

        activation = np.ones((m, n+1))
        activation[:,1:] = X[:,:]
        for theta in self._thetas:
            ZT = np.dot(activation, np.transpose(theta))
            Z_shape = ZT.shape
            AT = np.ones((Z_shape[0], Z_shape[1]+1))
            AT[:,1:] = self._math_helper.sigmoid(ZT)
            activation = AT

        return activation[:,1:]

    def predict_binary_classification(self, X_in, positive_threshold_inclusive = 0.5):
        hypothesis = self.evaluate(X_in)
        if hypothesis.shape[1] > 1:
            raise(ValueError("Output layer must have only one feature for binary classification, but found {0} features.".format(hypothesis.shape[1])))

        return np.asmatrix((hypothesis >= positive_threshold_inclusive).astype(int))


    def predict_multiclass_classification(self, X_in):
        hypothesis = self.evaluate(X_in)
        if hypothesis.shape[1] == 1:
            raise(ValueError("Found only 1 feature in output layer. Please use binary classifier `model.predict_binary_classification(...)`."))
        elif hypothesis.shape[1] == 2:
            raise(ValueError("Found 2 features in output layer. Please consider using binary classifier with just one output feature."))

        return np.asmatrix(hypothesis).argmax(1)