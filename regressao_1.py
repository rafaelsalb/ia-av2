from matplotlib import pyplot as plt
import numpy as np
from models.adaline import Adaline
from utils import standardize


n = 1000


def main():
    data = np.loadtxt("data/aerogerador.dat", delimiter="\t", dtype="float128")
    adaline = Adaline()
    X = data[:n, 0]
    X = X.reshape((X.shape[0], 1))
    print(X.shape)
    y = data[:n, 1]
    X_standardized = standardize(X)
    adaline.fit(X_standardized, y, 2, 0.1, 0.1)
    # print(adaline.w)
    # X_test = X[:]
    # y_test = adaline.predict(standardize(X_test))
    # assert y_test.shape == y.shape, f"y = {y.shape}, y_test = {y_test.shape}"

    x_plot = np.linspace(-1, 1, 1000)
    y_plot = adaline.predict(standardize(x_plot.reshape((x_plot.shape[0], 1))))
    plt.plot(x_plot, y_plot, label="Adaline")
    plt.scatter(X, y, label="Data")
    plt.show()

    # mean_square_error = (y_test - y) ** 2
    # mean_square_error = np.mean(mean_square_error)
    # print(f"Mean square error: {mean_square_error}")


if __name__ == "__main__":
    main()
