import numpy as np
import pytest
from scipy.io import savemat

from specrule.data import labeled_spectra, load_indian_pines


def test_loader_rejects_missing_key_and_spatial_mismatch(tmp_path):
    cube_path = tmp_path / "cube.mat"
    labels_path = tmp_path / "labels.mat"
    savemat(cube_path, {"wrong": np.zeros((2, 2, 3))})
    savemat(labels_path, {"indian_pines_gt": np.zeros((2, 2))})
    with pytest.raises(ValueError, match="indian_pines_corrected"):
        load_indian_pines(cube_path, labels_path)

    savemat(cube_path, {"indian_pines_corrected": np.zeros((2, 3, 4))})
    with pytest.raises(ValueError, match="spatial"):
        load_indian_pines(cube_path, labels_path)


def test_labeled_spectra_excludes_background():
    cube = np.arange(12, dtype=float).reshape(2, 2, 3)
    labels = np.array([[0, 1], [2, 0]])
    spectra, target = labeled_spectra(cube, labels)

    assert spectra.shape == (2, 3)
    assert target.tolist() == [1, 2]
