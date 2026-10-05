"""Calibration, out-of-distribution check and Grad-CAM add-ons."""
from pathlib import Path

import numpy as np
import pytest

from src.calibration import calibrate_probs, expected_calibration_error, fit_temperature, softmax
from src.ood import MahalanobisOOD

ROOT = Path(__file__).resolve().parents[1]
HAS_EXTRAS = (ROOT / "models" / "ood_mahalanobis.npz").exists() and (ROOT / "models" / "calibration.json").exists()
needs_extras = pytest.mark.skipif(not HAS_EXTRAS, reason="run python -m scripts.fit_extras first")


def test_temperature_scaling_keeps_predictions_and_reduces_overconfidence():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 5, 2000)
    logits = rng.normal(0, 1, (2000, 5))
    logits[np.arange(2000), y] += 2.0
    logits *= 4                                   # make the model over-confident
    t = fit_temperature(logits, y)
    raw, cal = softmax(logits), softmax(logits / t)
    assert t > 1 and (raw.argmax(1) == cal.argmax(1)).all()
    assert expected_calibration_error(cal, y) < expected_calibration_error(raw, y)
    assert np.allclose(calibrate_probs(raw, t), cal, atol=1e-6)


def test_mahalanobis_scores_far_points_higher():
    rng = np.random.default_rng(0)
    feats = np.r_[rng.normal(0, 1, (300, 8)), rng.normal(5, 1, (300, 8))]
    labels = np.r_[np.zeros(300, int), np.ones(300, int)]
    ood = MahalanobisOOD().fit(feats, labels)
    near, far = ood.score(rng.normal(0, 1, (50, 8))), ood.score(rng.normal(30, 1, (50, 8)))
    assert far.min() > near.max()


@needs_extras
def test_app_predictor_extras_on_a_leaf_and_a_non_leaf():
    from src.imaging import load_rgb
    from src.predictor import Predictor
    p = Predictor(device="cpu")
    leaf = load_rgb(sorted((ROOT / "assets" / "samples").glob("*.jpg"))[0])
    out = p.analyse(leaf)
    assert out["ood"] is False or out["ood"] == False                       # noqa: E712  (numpy bool)
    assert out["pred_index"] == int(np.argmax(p.calibrated(out["probs"])))   # calibration never changes the class
    # a real non-leaf photo (CIFAR-10) should be flagged; synthetic patterns are a known limitation, so not used here
    cifar = ROOT / "data" / "cifar10"
    if cifar.exists():
        from torchvision.datasets import CIFAR10
        from src.imaging import standardise
        car = standardise(CIFAR10(root=str(cifar), train=False, download=False)[0][0].convert("RGB"))
        assert p.ood_score(car) > p.ood.threshold
    cam = p.gradcam_overlay(out["standardised"], out["pred_index"])
    assert cam.size == (224, 224) and cam.mode == "RGB"
