"""Reliability features and the Random Forest that estimates P(prediction is incorrect).

Target: 1 = the classifier's prediction is INCORRECT (positive class), 0 = correct.
Configuration A uses only the uploaded image; B adds stability under a fixed perturbation policy
(len(stability_policy) extra forward passes per image, at training time AND in the app).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.model import ModelFileError

PROB_FEATURES = ["max_prob", "margin_top1_top2", "entropy_norm", "top3_mass"]
QUALITY_FEATURES = ["brightness", "contrast", "sharpness_log_lapvar", "noise_sigma", "saturation",
                    "original_min_side", "aspect_ratio"]
STABILITY_FEATURES = ["stability_agree_frac", "stability_mean_pred_prob", "stability_min_pred_prob",
                      "stability_std_pred_prob", "stability_mean_l1_shift", "stability_n_distinct"]
FEATURES = {"A": PROB_FEATURES + QUALITY_FEATURES, "B": PROB_FEATURES + QUALITY_FEATURES + STABILITY_FEATURES}


def prob_features(p: np.ndarray) -> dict[str, float]:
    s = np.sort(p)[::-1]
    ent = -(p * np.log(np.clip(p, 1e-12, 1))).sum() / np.log(len(p))
    return {"max_prob": float(s[0]), "margin_top1_top2": float(s[0] - s[1]),
            "entropy_norm": float(ent), "top3_mass": float(s[:3].sum())}


def stability_features(p: np.ndarray, perturbed: np.ndarray) -> dict[str, float]:
    """p: [C] probabilities of the image; perturbed: [K, C] probabilities of its K policy perturbations."""
    pred = int(p.argmax())
    pp = perturbed[:, pred]
    preds = perturbed.argmax(1)
    return {"stability_agree_frac": float((preds == pred).mean()),
            "stability_mean_pred_prob": float(pp.mean()),
            "stability_min_pred_prob": float(pp.min()),
            "stability_std_pred_prob": float(pp.std()),
            "stability_mean_l1_shift": float(np.abs(perturbed - p[None]).sum(1).mean()),
            "stability_n_distinct": float(len(set(preds.tolist()) | {pred}))}


def feature_vector(feats: dict, config: str) -> pd.DataFrame:
    """One-row table with the schema's columns in the schema's order - exactly what the forest was fitted on."""
    return pd.DataFrame([[float(feats[name]) for name in FEATURES[config]]], columns=FEATURES[config])


class ReliabilityModel:
    """Loads the saved Random Forests, schema and validation-selected thresholds."""

    def __init__(self, models_dir: Path):
        import joblib
        schema_path = models_dir / "reliability_schema.json"
        if not schema_path.exists():
            raise ModelFileError("Missing models/reliability_schema.json. Run the reliability step (evaluate_windows.bat).")
        self.schema = json.loads(schema_path.read_text())
        self.models = {}
        for cfg_name, info in self.schema["configs"].items():
            path = models_dir / info["model_file"]
            if not path.exists():
                raise ModelFileError(f"Missing reliability model: models/{info['model_file']}.")
            # joblib uses pickle: only load model files produced by this project
            self.models[cfg_name] = joblib.load(path)
            assert info["features"] == FEATURES[cfg_name], f"Feature schema mismatch for configuration {cfg_name}"

    def risk(self, feats: dict, config: str) -> float:
        return float(self.models[config].predict_proba(feature_vector(feats, config))[0, 1])

    def threshold(self, config: str) -> float:
        return float(self.schema["configs"][config]["threshold"])
