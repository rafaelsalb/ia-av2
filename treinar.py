import numpy as np
from multiprocessing import Pool
from evaluator import Evaluator
from models.mlp import MultilayerPerceptron
from utils import export_mlp, export_multilayer_weights


def train_model(model, X, y, epochs, tol, patience=5, patience_start=1000, key=None):
    _X = np.copy(X)
    _y = np.copy(y)

    train_size = int(_X.shape[1] * 0.75)
    X_train = _X[:, :train_size]
    X_test = _X[:, train_size:]
    y_train = _y[:, :train_size]
    y_test = _y[:, train_size:]

    model.train(X_train, y_train, epochs, tol, patience=patience, patience_start=patience_start)
    evaluator = Evaluator(model)
    results = evaluator.evaluate(X_test, y_test)
    return model, results

def main():
    topologies = [
        [4, 8, 16, 32, 16, 8, 4],
        [10 for _ in range(3)],
        [10 for _ in range(5)],
        [10 for _ in range(10)],
        [23 for _ in range(12)],
        [3 for _ in range(20)],
        [50 for _ in range(10)],
        [100 for _ in range(5)],
        [100 for _ in range(4)] + [4, 3, 2],
    ]
    models = [
        MultilayerPerceptron(
            p=3,
            q=topology,
            m=1,
            learning_rate=0.25
        )
        for topology in topologies
    ]

    data = np.loadtxt("data/Spiral3d.csv", delimiter=",")
    X = data[:, :3]
    y = data[:, 3:]
    X_s = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    X_s = X_s.T
    y = y.T

    epochs = 20_000
    tol = 0.01

    with Pool(processes=len(models)) as pool:
        results = pool.starmap(train_model, [(model, X_s, y, epochs, tol, 200, 2000, i) for i, model in enumerate(models)])

    for i, (model, results) in enumerate(results):
        export_mlp(model, f"weights/mlp_Spiral3d_{i}", results)
        print(f"Model {i} trained with topology {topologies[i]}")


if __name__ == "__main__":
    main()
