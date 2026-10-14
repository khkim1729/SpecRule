"""Numerically safe spectral rule features."""

import numpy as np


def _safe_denominator(value: np.ndarray, eps: float) -> np.ndarray:
    return np.where(np.abs(value) < eps, np.where(value < 0, -eps, eps), value)


def ratio(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Return a safe elementwise spectral ratio ``a / b``."""
    left, right = np.broadcast_arrays(np.asarray(a, dtype=float), np.asarray(b, dtype=float))
    return left / _safe_denominator(right, eps)


def ndsi(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Return the normalized difference spectral index ``(a-b)/(a+b)``."""
    left, right = np.broadcast_arrays(np.asarray(a, dtype=float), np.asarray(b, dtype=float))
    return (left - right) / _safe_denominator(left + right, eps)
