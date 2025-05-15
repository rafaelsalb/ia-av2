from itertools import product
import json
import numpy as np

from evaluator import Evaluator
from models.mlp import MultilayerPerceptron
from models.perceptron import SimplePerceptron
from utils import export_mlp, export_multilayer_weights, export_weights, one_hot_encode, standardize


data = np.loadtxt("data/Spiral3d.csv", delimiter=",")
X = data[:, :3]
y = data[:, 3:]

N, p = X.shape

train_size = int(N * 0.75)

X_s = standardize(X)
X_s = X_s.T
X_train = X_s[:, :train_size]
X_test = X_s[:, train_size:]

y = y.T
d = y
# y = one_hot_encode(y, axis=0)  # Classe 1: -1, Classe 2: 1
y_train = y[:, :train_size]
y_test = y[:, train_size:]

print("y", y[:10])
print("d", d[:10])

# model = SimplePerceptron()
# model.fit(X_train, y_train, 100, 0.1, 0.1)

# print("Exporting weights...")
# export_weights(model.w, "weights/perceptron_spiral3d.csv")
# print("Exported weights")

# result = model.predict(X_test)

# accuracy = np.sum(result == y_test) / N
# print(f"Accuracy: {accuracy * 100:.2f}%")
# print(result[:10])

m = 1

epochs = [
    100,
    1000,
    10_000,
    100_000,
]
qs = [
    [10, 10, 10],
    [23 for _ in range(23)],
    [50 for _ in range(10)]
]
learning_rates = [
    0.1,
    0.75,
    0.05,
    0.025,
    0.01
]
tols = [
    0.1,
    0.01,
    0.001,
]

d_mapped = d
d_mapped[d_mapped == -1] = 0
d_mapped[d_mapped == 1] = 1

hyperparams = product(epochs, qs, learning_rates, tols)

for epoch, q, learning_rate, tol in hyperparams:
    print(f"q: {q}, learning_rate: {learning_rate}, tol: {tol}")
    model = MultilayerPerceptron(p, q, m, learning_rate=learning_rate)
    model.train(X_train, y_train, epoch, learning_rate, tol)
    evaluator = Evaluator(model)
    accuracy, precision, recall = evaluator.evaluate(X_test, d_mapped[:, train_size:])
    result = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
    }
    # Export evaluation metrics
    with open(f"evaluation/mlp_spiral3d_{epoch}_{q}_{learning_rate}_{tol}.json", "w") as f:
        json.dump(result, f)
    print("Exporting weights...")
    export_mlp(model, f"weights/mlp_spiral3d_{epoch}_{q}_{learning_rate}_{tol}")
