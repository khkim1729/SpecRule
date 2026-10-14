"""Diagnostic visualizations for discovered rules."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .features import ndsi, ratio


def plot_best_rule(
    spectra: np.ndarray, labels: np.ndarray, rules: pd.DataFrame, output_path: str | Path
) -> Path:
    """Plot the highest-F1 rule's class-versus-rest value distributions."""
    score_columns = (
        ["heldout_f1", "heldout_accuracy"]
        if "heldout_f1" in rules
        else ["f1", "accuracy"]
    )
    row = rules.sort_values(score_columns, ascending=False).iloc[0]
    i, j = int(row.band_i) - 1, int(row.band_j) - 1
    feature_fn = ratio if row.feature == "ratio" else ndsi
    values = feature_fn(spectra[:, i], spectra[:, j])
    positive = labels == int(row.class_id)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.0), constrained_layout=True)
    ax.hist(values[~positive], bins=60, density=True, alpha=0.55, color="#7f8c8d", label="Other classes")
    ax.hist(values[positive], bins=40, density=True, alpha=0.7, color="#256399", label=str(row.class_name))
    ax.axvline(float(row.threshold), color="#d67e2c", linestyle="--", linewidth=2, label=f"Rule {row.direction} {row.threshold:.3f}")
    ax.set(xlabel=f"{row.feature.upper()}(Band {int(row.band_i)}, Band {int(row.band_j)})", ylabel="Density", title="Best held-out spectral rule")
    ax.legend(frameon=False)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output
