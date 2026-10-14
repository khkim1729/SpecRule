import numpy as np
import pandas as pd
from unittest.mock import patch

from specrule.search import ThresholdRule, discover_class_rules, find_best_threshold


def test_find_best_threshold_discovers_greater_direction():
    rule = find_best_threshold(np.array([0.0, 0.1, 0.9, 1.0]), np.array([0, 0, 1, 1]))

    assert rule.direction == ">"
    np.testing.assert_array_equal(rule.predict(np.array([0.0, 0.1, 0.9, 1.0])), [0, 0, 1, 1])
    assert rule.f1 == 1.0


def test_find_best_threshold_discovers_less_equal_direction():
    rule = find_best_threshold(np.array([0.0, 0.1, 0.9, 1.0]), np.array([1, 1, 0, 0]))

    assert rule.direction == "<="
    np.testing.assert_array_equal(rule.predict(np.array([0.0, 0.1, 0.9, 1.0])), [1, 1, 0, 0])
    assert rule.f1 == 1.0


def test_find_best_threshold_constructs_only_the_winning_rule():
    values = np.linspace(0.0, 1.0, 200)
    target = (values > 0.6).astype(int)

    with patch("specrule.search.ThresholdRule", wraps=ThresholdRule) as constructor:
        find_best_threshold(values, target)

    assert constructor.call_count == 1


def test_discovery_is_reproducible_and_returns_one_rule_per_nonzero_class():
    spectra = np.array(
        [[1, 4, 2], [2, 4, 1], [4, 1, 2], [5, 1, 1], [3, 2, 2]], dtype=float
    )
    labels = np.array([0, 1, 1, 2, 2])

    first = discover_class_rules(spectra, labels, max_pairs=3)
    second = discover_class_rules(spectra, labels, max_pairs=3)

    pd.testing.assert_frame_equal(first, second)
    assert first["class_id"].tolist() == [1, 2]
    assert len(first) == 2
