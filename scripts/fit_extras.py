"""Fit the two add-ons WITHOUT retraining the classifier, then evaluate them once on the test set.

  * calibration: temperature T fitted on VALIDATION logits
  * out-of-distribution check: Mahalanobis class means/covariance on TRAINING features, threshold = 99th percentile
    of VALIDATION scores (about 1% of genuine validation leaves rejected)
  * evaluation (reported only): PlantVillage test set, CIFAR-10 test images (clearly not leaves), PlantDoc field photos

    python -m scripts.fit_extras
"""
import json
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

from src.calibration import calibrate_probs, expected_calibration_error, fit_temperature, softmax
from src.config import get_device, load_config, project_path, set_seed
from src.datasets import ManifestDataset
from src.imaging import eval_transform, standardise
from src.model import features_and_logits, load_classifier
from src.ood import MahalanobisOOD
from src.plots import GREEN, PALETTE, plt

MET, FIG, MODELS = project_path("reports/metrics"), project_path("reports/figures"), project_path("models")
OOD_PERCENTILE = 99


class PILDataset(Dataset):
    """Wraps PIL images (CIFAR-10) through the same standardise + eval transform as uploads."""
    def __init__(self, images, cfg):
        self.images, self.cfg, self.t = images, cfg, eval_transform(cfg)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        return self.t(standardise(self.images[i].convert("RGB"), self.cfg["image"]["standard_size"])), 0


CACHE = project_path("data") / "feature_cache"     # outside the deliverable (data/ is git-ignored)


def extract(model, device, dataset, bs=64, workers=2, name=None):
    """Features + logits; cached per split so an interrupted run resumes instead of starting over."""
    cache = CACHE / f"{name}.npz" if name else None
    if cache is not None and cache.exists():
        d = np.load(cache)
        return d["feats"], d["logits"]
    feats, logits = [], []
    for i, (x, _) in enumerate(DataLoader(dataset, batch_size=bs, num_workers=workers)):
        f, lg = features_and_logits(model, x, device)
        feats.append(f)
        logits.append(lg)
        if i % 100 == 0:
            print(f"    {name or 'batch'}: {i * bs}/{len(dataset)}", flush=True)
    out = np.concatenate(feats), np.concatenate(logits)
    if cache is not None:
        CACHE.mkdir(parents=True, exist_ok=True)
        np.savez(cache, feats=out[0], logits=out[1])
    return out


def main():
    cfg = load_config()
    set_seed(cfg["seed"])
    device = get_device()
    model, class_names = load_classifier(MODELS, device)
    idx = {c: i for i, c in enumerate(class_names)}
    man = pd.read_csv(project_path("reports/manifests/plantvillage_splits.csv"))
    root = project_path(cfg["paths"]["plantvillage_dir"])
    t0 = time.time()

    data = {}
    for split in ("train", "val", "test"):
        d = man[man["split"] == split]
        ds = ManifestDataset(root, d["path"].tolist(), d["class_name"].map(idx).tolist(), eval_transform(cfg))
        data[split] = (*extract(model, device, ds, name=split), d["class_name"].map(idx).to_numpy())
        print(f"{split}: {len(d)} images ({time.time() - t0:.0f}s)", flush=True)

    # --- calibration (validation only) ---
    _, val_logits, val_y = data["val"]
    T = fit_temperature(val_logits, val_y)
    _, test_logits, test_y = data["test"]
    raw, cal = softmax(test_logits), softmax(test_logits / T)
    assert (raw.argmax(1) == cal.argmax(1)).all(), "temperature scaling must not change predictions"
    cal_metrics = {"temperature": T, "fitted_on": "validation split logits", "evaluated_on": "locked test split (clean)",
                   "test_ece_before": expected_calibration_error(raw, test_y), "test_ece_after": expected_calibration_error(cal, test_y),
                   "test_nll_before": float(-np.log(raw[np.arange(len(test_y)), test_y] + 1e-12).mean()),
                   "test_nll_after": float(-np.log(cal[np.arange(len(test_y)), test_y] + 1e-12).mean()),
                   "test_mean_confidence_before": float(raw.max(1).mean()), "test_mean_confidence_after": float(cal.max(1).mean()),
                   "test_accuracy": float((raw.argmax(1) == test_y).mean())}
    (MODELS / "calibration.json").write_text(json.dumps({"temperature": T, "method": "temperature scaling on validation logits"}, indent=2))
    (MET / "calibration_metrics.json").write_text(json.dumps(cal_metrics, indent=2))
    print("calibration", {k: round(v, 5) if isinstance(v, float) else v for k, v in cal_metrics.items()}, flush=True)

    # --- out-of-distribution check (train fit, validation threshold) ---
    tr_f, _, tr_y = data["train"]
    ood = MahalanobisOOD().fit(tr_f, tr_y)
    val_scores = ood.score(data["val"][0])
    threshold = float(np.percentile(val_scores, OOD_PERCENTILE))
    ood.save(MODELS / "ood_mahalanobis.npz", threshold,
             {"method": "class-conditional Mahalanobis distance on EfficientNet penultimate features",
              "fitted_on": "training split features", "threshold_rule": f"{OOD_PERCENTILE}th percentile of validation scores"})

    from torchvision.datasets import CIFAR10
    cifar = CIFAR10(root=str(project_path("data") / "cifar10"), train=False, download=True)
    rng = np.random.default_rng(cfg["seed"])
    cifar_imgs = [cifar[int(i)][0] for i in rng.choice(len(cifar), 2000, replace=False)]
    cifar_f, _ = extract(model, device, PILDataset(cifar_imgs, cfg), workers=0, name="cifar10")

    pdm = pd.read_csv(project_path("reports/manifests/plantdoc_manifest.csv"))
    pdroot = project_path(cfg["paths"]["plantdoc_dir"])
    pd_ds = ManifestDataset(pdroot, pdm["path"].tolist(), [0] * len(pdm), eval_transform(cfg))
    pd_f, _ = extract(model, device, pd_ds, workers=0, name="plantdoc")

    s_test, s_cifar, s_pd = ood.score(data["test"][0]), ood.score(cifar_f), ood.score(pd_f)
    y = np.r_[np.zeros(len(s_test)), np.ones(len(s_cifar))]
    ood_metrics = {"threshold": threshold, "threshold_rule": f"{OOD_PERCENTILE}th percentile of validation scores",
                   "plantvillage_test_rejected": float((s_test > threshold).mean()),
                   "cifar10_rejected": float((s_cifar > threshold).mean()), "cifar10_images": len(s_cifar),
                   "auroc_plantvillage_vs_cifar10": float(roc_auc_score(y, np.r_[s_test, s_cifar])),
                   "plantdoc_field_photos_rejected": float((s_pd > threshold).mean()), "plantdoc_images": len(s_pd),
                   "note": "CIFAR-10 (vehicles, animals) stands in for clearly non-leaf uploads; PlantDoc are real leaves in field photos."}
    (MET / "ood_metrics.json").write_text(json.dumps(ood_metrics, indent=2))
    print("ood", {k: round(v, 4) if isinstance(v, float) else v for k, v in ood_metrics.items()}, flush=True)

    # --- figures ---
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
    for probs, name, col in ((raw, f"before (ECE {cal_metrics['test_ece_before']:.4f})", PALETTE[4]),
                             (cal, f"after T={T:.2f} (ECE {cal_metrics['test_ece_after']:.4f})", GREEN)):
        conf, ok = probs.max(1), probs.argmax(1) == test_y
        edges = np.linspace(0.5, 1, 11)
        mids, accs = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (conf > lo) & (conf <= hi)
            if m.sum() >= 5:
                mids.append(conf[m].mean())
                accs.append(ok[m].mean())
        ax[0].plot(mids, accs, "o-", color=col, label=name)
    ax[0].plot([0.5, 1], [0.5, 1], "--", color="#999", label="perfect")
    ax[0].set(title="Classifier calibration (test set)", xlabel="Displayed confidence", ylabel="Actual accuracy")
    ax[0].legend(fontsize=8)
    bins = np.linspace(0, np.percentile(np.r_[s_test, s_cifar, s_pd], 99.5), 60)
    for s, name, col in ((s_test, "PlantVillage test (leaves)", GREEN), (s_pd, "PlantDoc (field leaves)", PALETTE[3]),
                         (s_cifar, "CIFAR-10 (not leaves)", PALETTE[4])):
        ax[1].hist(np.clip(s, None, bins[-1]), bins=bins, alpha=0.55, color=col, label=name, density=True)
    ax[1].axvline(threshold, color="black", ls="--", label="rejection threshold")
    ax[1].set(title="Out-of-distribution score", xlabel="Mahalanobis distance (higher = more unusual)", ylabel="Density")
    ax[1].legend(fontsize=8)
    fig.savefig(FIG / "calibration_and_ood.png")
    print(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
