import numpy as np
import cv2
from skimage.feature import local_binary_pattern
from scipy.special import factorial
from numpy.linalg import pinv


# ============================================================
# Extreme Learning Machine
# ============================================================
class ELM:
    def __init__(self, input_dim, hidden_dim=1000):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

    def _activation(self, X):
        return 1 / (1 + np.exp(-X))

    def fit(self, X, y):
        # Random weights and bias
        self.W = np.random.randn(self.input_dim, self.hidden_dim)
        self.b = np.random.randn(self.hidden_dim)

        H = self._activation(X @ self.W + self.b)
        self.beta = pinv(H) @ y

    def predict(self, X):
        H = self._activation(X @ self.W + self.b)
        return H @ self.beta
