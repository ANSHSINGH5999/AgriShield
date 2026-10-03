"""Batch engine: runs the classifier on evaluation conditions of each image and computes reliability features exactly
as the app does for an uploaded image (the condition image plays the role of the upload).

Plan per image (fixed by seed + image_id, so it is reproducible):
  * robustness conditions: clean + every corruption x severity (test set only) - one forward pass each
  * reliability conditions: clean + 3 corruption conditions sampled uniformly from the 18 - each also gets the
    6 stability-policy passes, so these rows carry the full feature set used by Random Forest A and B
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src import perturbations
from src.imaging import load_rgb, standardise
from src.model import predict_probs
from src.quality import quality_features
from src.reliability import STABILITY_FEATURES, prob_features, stability_features

RELIABILITY_SAMPLED_CONDITIONS = 3


def reliability_condition_indices(image_id: int, n_conditions: int, seed: int) -> list[int]:
    """Condition 0 (clean) plus RELIABILITY_SAMPLED_CONDITIONS distinct corrupted conditions, seeded per image."""
    rng = np.random.default_rng(seed * 1_000_003 + int(image_id))
    return [0] + sorted(rng.choice(np.arange(1, n_conditions), RELIABILITY_SAMPLED_CONDITIONS, replace=False).tolist())


class VariantDataset(Dataset):
    """Item = uint8 crops for every planned pass of one image, plus the layout and quality features."""

    def __init__(self, root: Path, items: pd.DataFrame, cfg: dict, conditions: list, all_conditions: bool):
        self.root, self.items, self.cfg, self.conditions = root, items, cfg, conditions
        self.all_conditions = all_conditions
        self.policy = cfg["stability_policy"]
        self.size, self.crop = cfg["image"]["standard_size"], cfg["image"]["input_size"]

    def __len__(self):
        return len(self.items)

    def _crop(self, img):
        w, h = img.size
        left, top = (w - self.crop) // 2, (h - self.crop) // 2   # identical to torchvision CenterCrop (verified in tests)
        a = np.asarray(img.crop((left, top, left + self.crop, top + self.crop)), dtype=np.uint8)
        return torch.from_numpy(a.copy()).permute(2, 0, 1)

    def __getitem__(self, i):
        row = self.items.iloc[i]
        std = standardise(load_rgb(self.root / row["path"]), self.size)
        rel = set(reliability_condition_indices(row["image_id"], len(self.conditions), self.cfg["seed"]))
        todo = range(len(self.conditions)) if self.all_conditions else sorted(rel)
        crops, layout, quals = [], [], {}
        for c_i in todo:
            fam, sev, val = self.conditions[c_i]
            cond = perturbations.apply_condition(std, fam, val, seed=int(row["image_id"]) * 100 + c_i)
            crops.append(self._crop(cond))
            with_stab = c_i in rel
            if with_stab:
                quals[c_i] = quality_features(cond, cond.size)
                for k, (pf, pv) in enumerate(self.policy):          # identical to Predictor.policy_images
                    crops.append(self._crop(perturbations.apply(cond, pf, pv, seed=k)))
            layout.append((c_i, with_stab))
        return torch.stack(crops), layout, quals


def _collate(batch):
    return torch.cat([b[0] for b in batch]), [b[1] for b in batch], [b[2] for b in batch]


def run(model, device, cfg, root: Path, items: pd.DataFrame, class_index: dict, conditions: list, all_conditions: bool,
        batch_images: int = 6, workers: int = 6) -> pd.DataFrame:
    """items: DataFrame with path, class_name, image_id. Returns one row per (image, planned condition)."""
    items = items.reset_index(drop=True)
    dl = DataLoader(VariantDataset(root, items, cfg, conditions, all_conditions), batch_size=batch_images,
                    num_workers=workers, collate_fn=_collate)
    mean = torch.tensor(cfg["image"]["mean"], device=device).view(1, 3, 1, 1)
    std = torch.tensor(cfg["image"]["std"], device=device).view(1, 3, 1, 1)
    k = len(cfg["stability_policy"])
    rows, img_i, t_start, next_log = [], 0, time.time(), 250
    for crops, layouts, quals in dl:
        x = (crops.to(device).float() / 255.0 - mean) / std
        t0 = time.time()
        probs = predict_probs(model, x, device)
        ms_per_pass = (time.time() - t0) * 1000 / len(x)
        pos = 0
        for layout, qual in zip(layouts, quals):
            meta = items.iloc[img_i]
            true = class_index[meta["class_name"]]
            per_cond = {}
            for c_i, with_stab in layout:
                p = probs[pos]
                pert = probs[pos + 1: pos + 1 + k] if with_stab else None
                pos += 1 + (k if with_stab else 0)
                per_cond[c_i] = (p, pert)
            clean_p = per_cond[0][0]
            for c_i, (p, pert) in per_cond.items():
                fam, sev, val = conditions[c_i]
                pred = int(p.argmax())
                row = {"image_id": int(meta["image_id"]), "path": meta["path"], "true_index": true,
                       "family": fam, "severity": sev, "value": val, "has_reliability_features": int(pert is not None),
                       "clean_pred": int(clean_p.argmax()), "clean_conf": float(clean_p.max()),
                       "pred_index": pred, "confidence": float(p[pred]), "correct": int(pred == true),
                       "incorrect": int(pred != true), "changed_vs_clean": int(pred != int(clean_p.argmax())),
                       "inference_ms_per_pass": ms_per_pass, **prob_features(p)}
                if pert is not None:
                    row.update(qual[c_i])
                    row.update(stability_features(p, pert))
                else:
                    row.update({f: np.nan for f in STABILITY_FEATURES})
                rows.append(row)
            img_i += 1
        if img_i >= next_log:
            rate = img_i / (time.time() - t_start)
            print(f"  {img_i}/{len(items)} images ({rate:.1f} img/s, ~{(len(items) - img_i) / rate / 60:.0f} min left)", flush=True)
            next_log += 250
    return pd.DataFrame(rows)
