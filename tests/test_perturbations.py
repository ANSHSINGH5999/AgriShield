import numpy as np
import pytest

from src import perturbations
from src.config import load_config
from src.imaging import standardise


@pytest.mark.parametrize("family", perturbations.FAMILIES)
def test_every_family_keeps_size_and_mode(leaf_image, family):
    img = standardise(leaf_image)
    val = load_config()["perturbations"][family][1]
    out = perturbations.apply(img, family, val, seed=1)
    assert out.size == img.size and out.mode == "RGB"
    assert not np.array_equal(np.asarray(out), np.asarray(img))


def test_brightness_direction(leaf_image):
    img = standardise(leaf_image)
    m = np.asarray(img).mean()
    assert np.asarray(perturbations.apply(img, "brightness_low", 0.5)).mean() < m
    assert np.asarray(perturbations.apply(img, "brightness_high", 1.5)).mean() > m


def test_noise_is_deterministic_for_a_seed(leaf_image):
    img = standardise(leaf_image)
    a = perturbations.apply(img, "gaussian_noise", 0.1, seed=3)
    b = perturbations.apply(img, "gaussian_noise", 0.1, seed=3)
    assert np.array_equal(np.asarray(a), np.asarray(b))


def test_conditions_cover_clean_and_all_severities():
    conds = perturbations.conditions(load_config())
    assert conds[0][0] == "clean" and len(conds) == 1 + 6 * 3


def test_unknown_family_raises(leaf_image):
    with pytest.raises(ValueError):
        perturbations.apply(leaf_image, "fog", 1)
