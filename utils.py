import json
import numpy as np


def standardize(X: np.ndarray) -> np.ndarray:
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_std = (X - mean) / std
    return X_std

def normalize(X: np.ndarray) -> np.ndarray:
    """
    Normalize the input data X to the range [0, 1].
    """
    X_min = np.min(X)
    X_max = np.max(X)
    X_range = X_max - X_min
    X_normalized = (X - X_min) / X_range
    return X_normalized

def export_weights(w: np.ndarray, filename: str) -> None:
    np.savetxt(filename, w, delimiter=",")

def import_weights(filename: str) -> np.ndarray:
    w = np.loadtxt(filename, delimiter=",")
    return w

def export_multilayer_weights(layers: list, filename: str, results: dict) -> None:
    """
    Export the weights of a multilayer perceptron to a CSV file.
    Each layer's weights are saved in a separate CSV file.
    """
    with open(filename + "_hyperparameters.json", "w") as f:
        model_hyperparameters = {
            "p": layers[0].w.shape[0],
            "q": [layer.w.shape[1] for layer in layers],
            "m": layers[0].w.shape[1],
            "learning_rate": layers[0].learning_rate,
            "weights": [layer.w.tolist() for layer in layers],
            "results": results
        }
        json.dump(model_hyperparameters, f)

def import_multilayer_weights(layers: list, filename: str) -> None:
    """
    Import the weights of a multilayer perceptron from CSV files.
    Each layer's weights are loaded from a separate CSV file.
    """
    for i, layer in enumerate(layers):
        layer_filename = f"{filename}_layer_{i}.csv"
        layer.w = np.loadtxt(layer_filename, delimiter=",")
        print(f"Imported weights of layer {i} from {layer_filename}")

def export_mlp(model: object, filename: str, results: dict) -> None:
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

    with open(filename + "_results.json", "w") as f:
        json.dump(results, f)
    print(f"Exported results to {filename}_results.json")

    with open(filename + "_weights.json", "w") as f:
        model_weights = {
            "weights": [layer.w.tolist() for layer in model.layers]
        }
        json.dump(model_weights, f)
    print(f"Exported weights to {filename}_weights.json")

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
