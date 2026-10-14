"""Evaluation helpers for serialized spectral rules."""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def binary_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    """Return positive-class F1 and accuracy for binary predictions."""
    truth = np.asarray(target).astype(np.int8)
    guessed = np.asarray(prediction).astype(np.int8)
    return {
        "f1": float(f1_score(truth, guessed, zero_division=0)),
        "accuracy": float(accuracy_score(truth, guessed)),
    }
