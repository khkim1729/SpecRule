"""Deterministic one-threshold spectral rule search."""

from dataclasses import dataclass
from itertools import islice

import numpy as np
import pandas as pd

from .features import ndsi, ratio


CLASS_NAMES = {
    1: "Alfalfa",
    2: "Corn-notill",
    3: "Corn-mintill",
    4: "Corn",
    5: "Grass-pasture",
    6: "Grass-trees",
    7: "Grass-pasture-mowed",
    8: "Hay-windrowed",
    9: "Oats",
    10: "Soybean-notill",
    11: "Soybean-mintill",
    12: "Soybean-cleantill",
    13: "Wheat",
    14: "Woods",
    15: "Buildings-Grass-Trees-Drives",
    16: "Stone-Steel-Towers",
}


@dataclass(frozen=True)
class ThresholdRule:
    threshold: float
    direction: str
    f1: float
    accuracy: float

    def predict(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values)
        if self.direction == "<=":
            return (array <= self.threshold).astype(np.int8)
        if self.direction == ">":
            return (array > self.threshold).astype(np.int8)
        raise ValueError(f"unsupported direction: {self.direction}")


def _score(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray, tn: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    denominator = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denominator, out=np.zeros_like(tp, dtype=float), where=denominator != 0)
    return f1, (tp + tn) / (tp + fp + fn + tn)


def find_best_threshold(values: np.ndarray, target: np.ndarray) -> ThresholdRule:
    """Find the deterministic F1-optimal one-dimensional binary rule."""
    x = np.asarray(values, dtype=float).reshape(-1)
    y = np.asarray(target, dtype=np.int8).reshape(-1)
    if x.size != y.size or x.size < 2 or not np.isin(y, [0, 1]).all():
        raise ValueError("values and binary target must have equal length >= 2")
    if not np.isfinite(x).all():
        raise ValueError("rule values must be finite")
    order = np.argsort(x, kind="mergesort")
    sorted_x = x[order]
    sorted_y = y[order]
    boundaries = np.flatnonzero(sorted_x[:-1] < sorted_x[1:])
    if boundaries.size == 0:
        boundaries = np.array([x.size - 1])
        thresholds = np.array([sorted_x[-1]])
    else:
        thresholds = (sorted_x[boundaries] + sorted_x[boundaries + 1]) / 2.0

    cumulative = np.cumsum(sorted_y)
    total_positive = cumulative[-1]
    prefix_positive = cumulative[boundaries]
    prefix_size = boundaries + 1
    suffix_positive = total_positive - prefix_positive
    suffix_size = x.size - prefix_size

    le_f1, le_acc = _score(
        prefix_positive,
        prefix_size - prefix_positive,
        total_positive - prefix_positive,
        (x.size - total_positive) - (prefix_size - prefix_positive),
    )
    gt_f1, gt_acc = _score(
        suffix_positive,
        suffix_size - suffix_positive,
        total_positive - suffix_positive,
        (x.size - total_positive) - (suffix_size - suffix_positive),
    )
    all_thresholds = np.concatenate([thresholds, thresholds])
    all_f1 = np.concatenate([le_f1, gt_f1])
    all_accuracy = np.concatenate([le_acc, gt_acc])
    all_direction = np.concatenate(
        [np.zeros(thresholds.size, dtype=np.int8), np.ones(thresholds.size, dtype=np.int8)]
    )
    winner = np.lexsort(
        (-all_thresholds, all_direction, all_accuracy, all_f1)
    )[-1]
    return ThresholdRule(
        float(all_thresholds[winner]),
        ">" if all_direction[winner] else "<=",
        float(all_f1[winner]),
        float(all_accuracy[winner]),
    )


def _band_pairs(n_bands: int, max_pairs: int):
    candidate_count = min(n_bands, max(2, int(np.ceil(np.sqrt(2 * max_pairs))) + 1))
    candidates = np.unique(np.linspace(0, n_bands - 1, candidate_count, dtype=int))
    pairs = ((int(i), int(j)) for pos, i in enumerate(candidates) for j in candidates[pos + 1 :])
    return islice(pairs, max_pairs)


def discover_class_rules(
    spectra: np.ndarray, labels: np.ndarray, max_pairs: int = 256
) -> pd.DataFrame:
    """Return the best ratio or NDSI threshold rule for every nonzero class."""
    x = np.asarray(spectra, dtype=float)
    y = np.asarray(labels).reshape(-1)
    if x.ndim != 2 or x.shape[0] != y.size:
        raise ValueError("spectra must be 2-D and aligned with labels")
    mask = y > 0
    x, y = x[mask], y[mask]
    if x.shape[0] < 2 or max_pairs < 1:
        raise ValueError("at least two labeled spectra and one pair are required")

    best: dict[int, tuple[ThresholdRule, str, int, int]] = {}
    classes = np.unique(y).astype(int)
    for band_i, band_j in _band_pairs(x.shape[1], max_pairs):
        for feature_name, values in (
            ("ratio", ratio(x[:, band_i], x[:, band_j])),
            ("ndsi", ndsi(x[:, band_i], x[:, band_j])),
        ):
            for class_id in classes:
                rule = find_best_threshold(values, (y == class_id).astype(np.int8))
                current = best.get(class_id)
                candidate_key = (rule.f1, rule.accuracy, -band_i, -band_j, feature_name == "ratio")
                current_key = (
                    (current[0].f1, current[0].accuracy, -current[2], -current[3], current[1] == "ratio")
                    if current
                    else None
                )
                if current_key is None or candidate_key > current_key:
                    best[class_id] = (rule, feature_name, band_i, band_j)

    rows = []
    for class_id in classes:
        rule, feature_name, band_i, band_j = best[class_id]
        rows.append(
            {
                "class_id": class_id,
                "class_name": CLASS_NAMES.get(class_id, f"Class {class_id}"),
                "feature": feature_name,
                "band_i": band_i + 1,
                "band_j": band_j + 1,
                "threshold": rule.threshold,
                "direction": rule.direction,
                "f1": rule.f1,
                "accuracy": rule.accuracy,
            }
        )
    return pd.DataFrame(rows)
