import numpy as np


class Evaluator:
    def __init__(self, model):
        self.model = model

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate the model on the given dataset.

        Parameters:
        - X: Input features (numpy array).
        - y: True labels (numpy array).
        """
        predictions = self.model.predict(X)
        # Classe -1: [0, 1], Classe 1: [1, 0]
        true_positives = (predictions[0, :] == 0) & (y[0, :] == 1)
        false_positives = (predictions[0, :] == 0) & (y[0, :] == 0)
        true_negatives = (predictions[0, :] == 1) & (y[0, :] == 0)
        false_negatives = (predictions[0, :] == 1) & (y[0, :] == 1)
        accuracy = (np.sum(true_positives) + np.sum(true_negatives)) / y.shape[1]
        precision = np.sum(true_positives) / (np.sum(true_positives) + np.sum(false_positives))
        recall = np.sum(true_positives) / (np.sum(true_positives) + np.sum(false_negatives))
        return accuracy, precision, recall
