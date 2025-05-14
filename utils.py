import json
import numpy as np


def standardize(X: np.ndarray) -> np.ndarray:
    mean = np.mean(X)
    std = np.std(X)
    X = (X - mean) / std
    return X

def export_weights(w: np.ndarray, filename: str) -> None:
    np.savetxt(filename, w, delimiter=",")

def import_weights(filename: str) -> np.ndarray:
    w = np.loadtxt(filename, delimiter=",")
    return w

def export_multilayer_weights(layers: list, filename: str) -> None:
    """
    Export the weights of a multilayer perceptron to a CSV file.
    Each layer's weights are saved in a separate CSV file.
    """
    for i, layer in enumerate(layers):
        layer_filename = f"{filename}_layer_{i}.csv"
        np.savetxt(layer_filename, layer.w, delimiter=",")
        print(f"Exported weights of layer {i} to {layer_filename}")

def import_multilayer_weights(layers: list, filename: str) -> None:
    """
    Import the weights of a multilayer perceptron from CSV files.
    Each layer's weights are loaded from a separate CSV file.
    """
    for i, layer in enumerate(layers):
        layer_filename = f"{filename}_layer_{i}.csv"
        layer.w = np.loadtxt(layer_filename, delimiter=",")
        print(f"Imported weights of layer {i} from {layer_filename}")

def export_mlp(model: object, filename: str) -> None:
    """
    Export the weights of a multilayer perceptron to a CSV file.
    Each layer's weights are saved in a separate CSV file.
    """
    model_hyperparameters = {
        "p": model.p,
        "q": model.q,
        "m": model.m,
        "learning_rate": model.learning_rate,
    }
    with open(filename + "_hyperparameters.json", "w") as f:
        json.dump(model_hyperparameters, f)
    print(f"Exported hyperparameters to {filename}_hyperparameters.json")

    for i, layer in enumerate(model.layers):
        layer_filename = f"{filename}_layer_{i}.csv"
        np.savetxt(layer_filename, layer.w, delimiter=",")
        print(f"Exported weights of layer {i} to {layer_filename}")

def import_mlp(filename: str) -> tuple:
    """
    Import the hyperparameters and weights of a multilayer perceptron from CSV files.
    Each layer's weights are loaded from a separate CSV file.
    """
    with open(filename + "_hyperparameters.json", "r") as f:
        model_hyperparameters = json.load(f)

    p = model_hyperparameters["p"]
    q = model_hyperparameters["q"]
    m = model_hyperparameters["m"]
    learning_rate = model_hyperparameters["learning_rate"]

    layers = []
    for i in range(len(q) + 1):
        layer_filename = f"{filename}_layer_{i}.csv"
        w = np.loadtxt(layer_filename, delimiter=",")
        layers.append(w)

    return p, q, m, learning_rate, layers

def one_hot_encode(y: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    One-hot encode the target variable y.

    axis: int, default=0
        Axis along which to one-hot encode. 0 for rows, 1 for columns.
    """
    if axis == 0:
        y = y.T
    C = np.flip(np.unique(y))
    n_classes, N = C.shape[0], y.shape[0]
    print(C)
    print("n_classes", n_classes)
    one_hot = np.zeros((n_classes, N))
    print("one_hot", one_hot.shape)
    for i in range(y.shape[0]):
        j = np.where(C == y[i])[0][0]
        # print("j", j, "y[i]", y[i], "i", i)
        one_hot[j, i] = 1
    assert one_hot.shape[1] == N, f"one_hot: {one_hot.shape}, y: {y.shape}"
    assert one_hot.shape[0] == n_classes, f"one_hot: {one_hot.shape}, n_classes: {n_classes}"
    return one_hot
