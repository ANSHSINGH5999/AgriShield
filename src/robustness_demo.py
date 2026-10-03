"""Automatic robustness demonstration for the live app (presentation only - not used by the reliability model).

Uses the existing perturbation functions and the existing severity-1 ("mild") strengths from config.yaml, applied one at a
time to the standardised image, exactly like the offline robustness evaluation. Severity 1 is used because at severity 2 the
noise level alone drops measured test accuracy to about 20%, so nearly every image would fail and the final assessment would
not distinguish one image from another. All three severities remain in the offline evaluation (Model Performance page).
"""
from PIL import Image

from src import perturbations
from src.imaging import standardise

# (display label, perturbation family) - strengths come from config.yaml, severity 1 of 3
DEMO_FAMILIES = [("Brightness change", "brightness_low"), ("Blur", "gaussian_blur"), ("Gaussian noise", "gaussian_noise"),
                 ("JPEG compression", "jpeg_compression"), ("Low resolution", "low_resolution")]
DEMO_SEVERITY = 1


def demo_conditions(cfg: dict) -> list[tuple[str, str, float]]:
    return [(label, fam, float(cfg["perturbations"][fam][DEMO_SEVERITY - 1])) for label, fam in DEMO_FAMILIES]


def run(predictor, raw: Image.Image) -> list[dict]:
    """Original + each degraded version -> EfficientNet prediction and confidence (one batched forward pass)."""
    std = standardise(raw, predictor.cfg["image"]["standard_size"])
    conds = demo_conditions(predictor.cfg)
    images = [std] + [perturbations.apply(std, fam, val, seed=0) for _, fam, val in conds]
    probs = predictor.probs(images)
    original = int(probs[0].argmax())
    rows = []
    for (label, fam, val), p in zip([("Original", "clean", 0.0)] + conds, probs):
        pred = int(p.argmax())
        rows.append({"condition": label, "family": fam, "value": val, "pred_index": pred,
                     "pred_class": predictor.class_names[pred], "confidence": float(p[pred]),
                     "same_as_original": pred == original})
    return rows
