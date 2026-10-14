import numpy as np

from specrule.features import ndsi, ratio


def test_ratio_and_ndsi_are_finite_with_zero_denominators():
    a = np.array([0.0, 1.0, -1.0])
    b = np.zeros(3)

    assert np.isfinite(ratio(a, b)).all()
    assert np.isfinite(ndsi(a, b)).all()


def test_ndsi_is_bounded_for_nonnegative_reflectance():
    a = np.array([0.0, 1.0, 2.0, 4.0])
    b = np.array([4.0, 2.0, 1.0, 0.0])
    values = ndsi(a, b)

    assert values.min() >= -1.0
    assert values.max() <= 1.0
