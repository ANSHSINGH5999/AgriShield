"""One object that turns an uploaded image into: prediction, top-3, confidence, quality indicators,
stability and reliability risk. Used by the app, the inference check and the tests."""
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src import perturbations
from src.config import get_device, load_config, project_path
from src.imaging import standardise, to_model_input, eval_transform
from src.model import load_classifier, predict_probs
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
