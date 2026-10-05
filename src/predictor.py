"""One object that turns an uploaded image into: prediction, top-3, confidence, quality indicators,
stability and reliability risk. Used by the app, the inference check and the tests."""
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src import perturbations
from src.config import get_device, load_config, project_path
from src.imaging import standardise, to_model_input, eval_transform
from src.calibration import calibrate_probs, load_temperature
from src.gradcam import gradcam, overlay
from src.model import features_and_logits, load_classifier, predict_probs
from src.ood import MahalanobisOOD
from src.quality import quality_features
from src.reliability import ReliabilityModel, prob_features, stability_features


def pretty(name: str) -> str:
    """'Tomato___Late_blight' -> 'Tomato — Late blight'."""
    plant, _, disease = name.partition("___")
    return f"{plant.replace('_', ' ').replace(',', '')} — {disease.replace('_', ' ').strip()}"


class Predictor:
    def __init__(self, models_dir: Path | None = None, device: str | None = None):
        self.cfg = load_config()
        self.models_dir = models_dir or project_path(self.cfg["paths"]["models_dir"])
        self.device = device or get_device()
        self.model, self.class_names = load_classifier(self.models_dir, self.device)
        self.transform = eval_transform(self.cfg)
        self.reliability = None
        if (self.models_dir / "reliability_schema.json").exists():
            self.reliability = ReliabilityModel(self.models_dir)
        self.temperature = load_temperature(self.models_dir)          # display-only calibration (None if not fitted)
        ood_path = self.models_dir / "ood_mahalanobis.npz"
        self.ood = MahalanobisOOD.load(ood_path) if ood_path.exists() else None

    def calibrated(self, probs: np.ndarray) -> np.ndarray:
        """Calibrated probabilities for DISPLAY. The reliability model always receives the raw probabilities."""
        return probs if self.temperature is None else calibrate_probs(probs, self.temperature)

    def ood_score(self, std_img: Image.Image) -> float:
        feats, _ = features_and_logits(self.model, self.transform(std_img).unsqueeze(0), self.device)
        return float(self.ood.score(feats)[0])

    def gradcam_overlay(self, std_img: Image.Image, class_index: int) -> Image.Image:
        x = self.transform(std_img).unsqueeze(0)
        size = self.cfg["image"]["input_size"]
        left, top = (std_img.width - size) // 2, (std_img.height - size) // 2      # same centre crop the model sees
        crop = std_img.crop((left, top, left + size, top + size))
        return overlay(crop, gradcam(self.model, x, class_index, self.device))

    def probs(self, images: list[Image.Image]) -> np.ndarray:
        batch = torch.stack([self.transform(im) for im in images])
        return predict_probs(self.model, batch, self.device)

    def policy_images(self, std_img: Image.Image) -> list[Image.Image]:
        return [perturbations.apply(std_img, fam, val, seed=i) for i, (fam, val) in enumerate(self.cfg["stability_policy"])]

    def analyse(self, raw: Image.Image, with_stability: bool = True) -> dict:
        """raw: RGB image as uploaded. Returns everything the app displays."""
        std = standardise(raw, self.cfg["image"]["standard_size"])
        imgs = [std] + (self.policy_images(std) if with_stability else [])
        all_p = self.probs(imgs)
        p = all_p[0]
        top = np.argsort(p)[::-1][:3]
        feats = {**prob_features(p), **quality_features(std, raw.size)}
        out = {"standardised": std, "probs": p, "pred_index": int(top[0]), "pred_class": self.class_names[top[0]],
               "confidence": float(p[top[0]]),
               "top3": [(self.class_names[i], float(p[i])) for i in top], "features": feats}
        out["display_confidence"] = float(self.calibrated(p)[top[0]])
        if self.ood is not None:
            out["ood_score"] = self.ood_score(std)
            out["ood"] = out["ood_score"] > self.ood.threshold
        if with_stability:
            feats.update(stability_features(p, all_p[1:]))
            out["stability"] = feats["stability_agree_frac"]
        if self.reliability is not None:
            cfg_name = "B" if with_stability else "A"
            out["risk"] = self.reliability.risk(feats, cfg_name)
            out["threshold"] = self.reliability.threshold(cfg_name)
            out["reliability_config"] = cfg_name
            out["flagged"] = out["risk"] >= out["threshold"]
        return out
