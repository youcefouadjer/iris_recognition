import numpy as np
import cv2
from skimage.feature import local_binary_pattern
from scipy.special import factorial
from numpy.linalg import pinv

# ============================================================
# Binary Bat Algorithm for feature selection
# ============================================================
class BBAFeatureSelector:
    def __init__(self, n_bats=20, n_iter=30, fmin=0, fmax=2):
        self.n_bats = n_bats
        self.n_iter = n_iter
        self.fmin = fmin
        self.fmax = fmax

    def _fitness(self, X, y, mask):
        """
        Fitness using ELM accuracy.
        """
        if mask.sum() == 0:
            return 1.0

        X_sel = X[:, mask == 1]

        # simple split
        split = int(0.8 * len(X_sel))
        Xtr, Xte = X_sel[:split], X_sel[split:]
        ytr, yte = y[:split], y[split:]

        elm = ELM(input_dim=X_sel.shape[1], hidden_dim=500)
        elm.fit(Xtr, ytr)
        pred = elm.predict(Xte)

        pred = np.argmax(pred, axis=1)
        acc = np.mean(pred == yte)

        return 1 - acc  # minimize

    def select(self, X, y):
        n_features = X.shape[1]

        # Initialize bats
        Xb = np.random.randint(0, 2, (self.n_bats, n_features))
        V = np.zeros_like(Xb, dtype=float)

        fitness = np.array([self._fitness(X, y, Xb[i]) for i in range(self.n_bats)])
        best_idx = np.argmin(fitness)
        best = Xb[best_idx].copy()

        for t in range(self.n_iter):
            for i in range(self.n_bats):
                beta = np.random.rand()
                f = self.fmin + (self.fmax - self.fmin) * beta

                # velocity update
                V[i] = V[i] + (Xb[i] - best) * f

                # sigmoid transfer
                S = 1 / (1 + np.exp(-V[i]))
                rand = np.random.rand(n_features)
                Xb[i] = (rand < S).astype(int)

                # evaluate
                fit = self._fitness(X, y, Xb[i])

                if fit < fitness[i]:
                    fitness[i] = fit

            best_idx = np.argmin(fitness)
            best = Xb[best_idx].copy()

        self.best_mask = best
        return best
