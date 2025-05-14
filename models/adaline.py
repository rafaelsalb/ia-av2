import numpy as np


class Adaline:
    def __init__(self, w: np.ndarray | None = None, scale: float | None = None):
        self.w: np.ndarray | None = w

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int, learning_rate: float, tol: float) -> None:
        N, p = X.shape
        X = X.T
        X = np.vstack(
            (-1 * np.ones((1, N)), X)
        )
        w = np.zeros((p + 1, 1))

        EQM1 = 1
        EQM2 = 0
        hist_eqm = []
        e = 0
        while e < epochs and abs(EQM1-EQM2) > tol:
            EQM1 = self._mean_square_error(X,y,w)
            hist_eqm.append(EQM1)
            for k in range(N):
                x_k = X[:,k].reshape(p+1,1)
                u_k = (w.T@x_k)[0,0]
                d_k = np.float32(y[k])
                e_k = d_k - u_k
                w = w + learning_rate * e_k * x_k
            e += 1
            EQM2 = self._mean_square_error(X,y,w)
        self.w = w

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.vstack(
            (-1 * np.ones((1, X.shape[0])), X.T)
        )
        return X.T @ self.w

    @staticmethod
    def _mean_square_error(X, y, w) -> np.float64:
        eqm = np.float64(0)
        p_1, N = X.shape
        for k in range(N):
            x_k = X[:, k].reshape(p_1, 1)
            u_k = (w.T @ x_k)[0, 0]
            d_k = np.float64(y[k])
            eqm += (d_k - u_k) ** 2
        return eqm / (N * 2)
