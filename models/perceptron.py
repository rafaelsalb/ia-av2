import numpy as np


class SimplePerceptron:
    def __init__(self, w: np.ndarray | None = None):
        self.w = w

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int, learning_rate: float, tol: float) -> None:
        p, N = X.shape
        bias = -np.ones((1, N))
        X = np.vstack((bias, X))
        self.w = np.zeros((p + 1, 1))
        for epoch in range(epochs):
            errors = 0
            for k in range(N):
                x_k = X[:, k].reshape(p + 1, 1)
                u_k = (self.w.T @ x_k)[0, 0]
                y_k = np.sign(u_k)
                d_k = float(y[0, k])
                if y_k != d_k:
                    errors += 1
                    self.w += learning_rate * (d_k - u_k) * x_k
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Errors = {errors}")
        else:
            print("Max epochs reached")
            print(f"Final errors = {errors}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        p, N = X.shape
        bias = -np.ones((1, N))
        X = np.vstack((bias, X))
        y = (self.w.T @ X)
        y = np.sign(y)
        y[y == 0] = 1
        return y
