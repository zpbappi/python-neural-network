import scipy as sp
import numpy as np


class MathHelper:
    @staticmethod
    def sigmoid(x):
        return 1. / (1 + sp.exp(-x))

    def sigmoid_grad(self, x):
        sig = self.sigmoid(x)
        return np.multiply(sig, 1 - sig)
