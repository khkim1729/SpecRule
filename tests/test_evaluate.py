import numpy as np
import pandas as pd

from specrule.evaluate import evaluate_rule_table


def test_evaluate_rule_table_adds_heldout_metrics_without_changing_rule():
    spectra = np.array(
        [[1.0, 4.0], [2.0, 4.0], [4.0, 2.0], [4.0, 1.0]]
    )
    labels = np.array([1, 1, 2, 2])
    rules = pd.DataFrame(
        [
            {
                "class_id": 1,
                "class_name": "Class 1",
                "feature": "ratio",
                "band_i": 1,
                "band_j": 2,
                "threshold": 0.5,
                "direction": "<=",
                "discovery_f1": 0.9,
                "discovery_accuracy": 0.8,
            }
        ]
    )

    evaluated = evaluate_rule_table(rules, spectra, labels)

    assert evaluated.loc[0, "heldout_f1"] == 1.0
    assert evaluated.loc[0, "heldout_accuracy"] == 1.0
    pd.testing.assert_frame_equal(
        evaluated.drop(columns=["heldout_f1", "heldout_accuracy"]), rules
    )
