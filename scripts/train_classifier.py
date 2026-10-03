"""Step 2: fine-tune EfficientNet-B0 on the PlantVillage TRAIN split; pick the best epoch on VAL.

    python -m scripts.train_classifier            (add --epochs 1 --limit 512 for a quick smoke test)
"""
import argparse
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from src.config import get_device, load_config, project_path, set_seed
from src.datasets import ManifestDataset
from src.imaging import eval_transform, train_transform
from src.model import build_model, save_classifier


def run_epoch(model, loader, device, loss_fn, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total, n, preds, ys = 0.0, 0, [], []
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = loss_fn(out, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += loss.item() * len(y)
            n += len(y)
            preds.append(out.argmax(1).cpu())
            ys.append(y.cpu())
    p, t = torch.cat(preds).numpy(), torch.cat(ys).numpy()
    return total / n, float((p == t).mean()), float(f1_score(t, p, average="macro"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--limit", type=int, help="use only N images per split (smoke test)")
    args = ap.parse_args()
    cfg = load_config()
    tc = cfg["training"]
    set_seed(cfg["seed"])
    device = get_device()
    root = project_path(cfg["paths"]["plantvillage_dir"])
    man = pd.read_csv(project_path("reports/manifests/plantvillage_splits.csv"))
    man = man[man["split"].isin(["train", "val"])]
    class_names = sorted(man["class_name"].unique())
    idx = {c: i for i, c in enumerate(class_names)}

    def subset(name):
        d = man[man["split"] == name]
        if args.limit:
            d = d.sample(min(args.limit, len(d)), random_state=cfg["seed"])
        return d

    tr, va = subset("train"), subset("val")
    loaders = {
        "train": DataLoader(ManifestDataset(root, tr["path"].tolist(), tr["class_name"].map(idx).tolist(), train_transform(cfg)),
                            batch_size=tc["batch_size"], shuffle=True, num_workers=tc["num_workers"], persistent_workers=tc["num_workers"] > 0),
        "val": DataLoader(ManifestDataset(root, va["path"].tolist(), va["class_name"].map(idx).tolist(), eval_transform(cfg)),
                          batch_size=tc["batch_size"] * 2, num_workers=tc["num_workers"], persistent_workers=tc["num_workers"] > 0),
    }
    # class imbalance: weight each class by 1/sqrt(frequency), normalised to mean 1
    counts = tr["class_name"].map(idx).value_counts().reindex(range(len(class_names)), fill_value=1).to_numpy()
    w = 1 / np.sqrt(counts)
    weights = torch.tensor(w / w.mean(), dtype=torch.float32, device=device)

    model = build_model(len(class_names), pretrained=True).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    eval_loss = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=tc["learning_rate"], weight_decay=tc["weight_decay"])
    epochs = args.epochs or tc["epochs"]
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    models_dir = project_path(cfg["paths"]["models_dir"])

    history, best, bad = [], -1.0, 0
    print(f"device={device} train={len(tr)} val={len(va)} classes={len(class_names)}", flush=True)
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tl, ta, tf = run_epoch(model, loaders["train"], device, loss_fn, opt)
        vl, vacc, vf = run_epoch(model, loaders["val"], device, eval_loss)
        sched.step()
        history.append({"epoch": epoch, "train_loss": tl, "train_acc": ta, "val_loss": vl, "val_acc": vacc,
                        "val_macro_f1": vf, "seconds": time.time() - t0})
        print(f"epoch {epoch}: train loss {tl:.4f} acc {ta:.4f} | val loss {vl:.4f} acc {vacc:.4f} macro-F1 {vf:.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        if vf > best:
            best, bad = vf, 0
            save_classifier(model, class_names, models_dir)
            history[-1]["saved"] = True
        else:
            bad += 1
            if bad >= tc["patience"]:
                print("early stopping", flush=True)
                break

    best_epoch = max(history, key=lambda h: h["val_macro_f1"])["epoch"]
    (project_path("reports/metrics/training_history.json")).write_text(json.dumps(
        {"history": history, "best_epoch": best_epoch, "selection": "highest validation macro-F1",
         "device": device, "class_weights": "1/sqrt(class frequency), mean-normalised",
         "train_images": len(tr), "val_images": len(va), "limit": args.limit}, indent=2))
    (models_dir / "preprocessing.json").write_text(json.dumps({
        "standardise_shorter_side": cfg["image"]["standard_size"], "center_crop": cfg["image"]["input_size"],
        "color_order": "RGB", "scale": "0-1 (ToTensor)", "mean": cfg["image"]["mean"], "std": cfg["image"]["std"],
        "resize_interpolation": "bicubic"}, indent=2))
    (models_dir / "model_metadata.json").write_text(json.dumps({
        "architecture": "torchvision efficientnet_b0, classifier replaced with Linear(1280, num_classes)",
        "pretrained_init": "ImageNet IMAGENET1K_V1", "num_classes": len(class_names), "best_epoch": best_epoch,
        "best_val_macro_f1": best, "seed": cfg["seed"], "torch_version": torch.__version__,
        "trained_on": "PlantVillage train split (reports/manifests/plantvillage_splits.csv)"}, indent=2))

    h = pd.DataFrame(history)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(h.epoch, h.train_loss, "o-", label="train"); ax[0].plot(h.epoch, h.val_loss, "o-", label="validation")
    ax[0].set(title="Loss", xlabel="Epoch", ylabel="Cross-entropy"); ax[0].legend()
    ax[1].plot(h.epoch, h.train_acc, "o-", label="train accuracy"); ax[1].plot(h.epoch, h.val_acc, "o-", label="val accuracy")
    ax[1].plot(h.epoch, h.val_macro_f1, "s--", label="val macro-F1"); ax[1].set(title="Accuracy / F1", xlabel="Epoch"); ax[1].legend()
    for a in ax:
        a.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(project_path("reports/figures/training_curves.png"), dpi=150)
    print(f"best epoch {best_epoch}, val macro-F1 {best:.4f}")


if __name__ == "__main__":
    main()
