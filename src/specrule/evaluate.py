"""Evaluation helpers for serialized spectral rules."""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from .features import ndsi, ratio


def binary_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    """Return positive-class F1 and accuracy for binary predictions."""
    truth = np.asarray(target).astype(np.int8)
    guessed = np.asarray(prediction).astype(np.int8)
    return {
        "f1": float(f1_score(truth, guessed, zero_division=0)),
        "accuracy": float(accuracy_score(truth, guessed)),
    }


def evaluate_rule_table(
    rules: pd.DataFrame, spectra: np.ndarray, labels: np.ndarray
) -> pd.DataFrame:
    """Evaluate already selected class-versus-rest rules on held-out spectra."""
    x = np.asarray(spectra, dtype=float)
    y = np.asarray(labels).reshape(-1)
    if x.ndim != 2 or x.shape[0] != y.size:
        raise ValueError("spectra must be 2-D and aligned with labels")

    evaluated = rules.copy()
    heldout_f1: list[float] = []
    heldout_accuracy: list[float] = []
    for row in evaluated.itertuples(index=False):
        i, j = int(row.band_i) - 1, int(row.band_j) - 1
        if i < 0 or j < 0 or i >= x.shape[1] or j >= x.shape[1]:
            raise ValueError("rule band index is outside the spectra")
        feature_fn = ratio if row.feature == "ratio" else ndsi
        values = feature_fn(x[:, i], x[:, j])
        prediction = values <= row.threshold if row.direction == "<=" else values > row.threshold
        metrics = binary_metrics(y == int(row.class_id), prediction)
        heldout_f1.append(metrics["f1"])
        heldout_accuracy.append(metrics["accuracy"])
    evaluated["heldout_f1"] = heldout_f1
    evaluated["heldout_accuracy"] = heldout_accuracy
    return evaluated
