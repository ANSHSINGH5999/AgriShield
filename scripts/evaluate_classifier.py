"""Step 4: classifier results on the locked TEST set - clean metrics and robustness to each degradation.
Uses reports/features/test_rows.csv.gz (produced once by compute_features --split test).

    python -m scripts.evaluate_classifier
"""
import json
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_recall_fscore_support

from src.config import get_device, load_config, project_path
from src.model import CHECKPOINT, load_classifier
from src.plots import PALETTE, plt

MET, FIG = project_path("reports/metrics"), project_path("reports/figures")


def inference_latency(models_dir, cfg) -> dict:
    out = {}
    for device in sorted({get_device(), "cpu"}):
        model, _ = load_classifier(models_dir, device)
        x = torch.randn(1, 3, cfg["image"]["input_size"], cfg["image"]["input_size"])
        with torch.no_grad():
            for _ in range(5):
                model(x.to(device))
            t0 = time.time()
            for _ in range(30):
                r = model(x.to(device))
                r.cpu()
        out[device] = (time.time() - t0) / 30 * 1000
    return out


def main():
    cfg = load_config()
    models_dir = project_path(cfg["paths"]["models_dir"])
    class_names = json.loads((models_dir / "class_names.json").read_text())
    rows = pd.read_csv(project_path("reports/features/test_rows.csv.gz"))
    clean = rows[rows["family"] == "clean"]
    y, p = clean["true_index"].to_numpy(), clean["pred_index"].to_numpy()
    labels = list(range(len(class_names)))
    pr, rc, f1, _ = precision_recall_fscore_support(y, p, labels=labels, average="macro", zero_division=0)
    rep = classification_report(y, p, labels=labels, target_names=class_names, output_dict=True, zero_division=0)
    cm = confusion_matrix(y, p, labels=labels)
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(MET / "test_confusion_matrix.csv")
    per_class = pd.DataFrame({c: rep[c] for c in class_names}).T.rename(columns={"support": "test_images"})
    per_class.to_csv(MET / "test_per_class_metrics.csv")
    latency = inference_latency(models_dir, cfg)
    clean_metrics = {
        "split": "PlantVillage locked test set (clean images)", "test_images": int(len(clean)),
        "accuracy": float(accuracy_score(y, p)), "macro_precision": float(pr), "macro_recall": float(rc), "macro_f1": float(f1),
        "num_classes": len(class_names), "class_distribution": clean["true_index"].map(lambda i: class_names[i]).value_counts().to_dict(),
        "model_file_mb": (models_dir / CHECKPOINT).stat().st_size / 1e6,
        "single_image_latency_ms": latency,
    }
    model, _ = load_classifier(models_dir, "cpu")
    clean_metrics["parameters"] = int(sum(q.numel() for q in model.parameters()))
    (MET / "classifier_test_metrics.json").write_text(json.dumps(clean_metrics, indent=2))
    (MET / "test_classification_report.txt").write_text(
        classification_report(y, p, labels=labels, target_names=class_names, digits=4, zero_division=0))

    # robustness: every corruption family x severity on the same test images
    summ = []
    for (fam, sev), g in rows.groupby(["family", "severity"]):
        summ.append({"family": fam, "severity": int(sev), "value": float(g["value"].iloc[0]), "images": len(g),
                     "accuracy": float(g["correct"].mean()),
                     "macro_f1": float(f1_score(g["true_index"], g["pred_index"], labels=labels, average="macro", zero_division=0)),
                     "prediction_change_rate": float(g["changed_vs_clean"].mean()),
                     "mean_confidence": float(g["confidence"].mean()),
                     "inference_ms_per_pass": float(g["inference_ms_per_pass"].mean())})
    rob = pd.DataFrame(summ).sort_values(["family", "severity"])
    rob.to_csv(MET / "robustness_by_corruption.csv", index=False)

    # figures
    fig, ax = plt.subplots(figsize=(11, 10))
    cmn = cm / cm.sum(1, keepdims=True).clip(min=1)
    im = ax.imshow(cmn, cmap="Greens", vmin=0, vmax=1)
    ax.set_xticks(labels, [c.replace("___", ": ")[:28] for c in class_names], rotation=90, fontsize=6)
    ax.set_yticks(labels, [c.replace("___", ": ")[:28] for c in class_names], fontsize=6)
    ax.set(title=f"Confusion matrix, clean test set (row-normalised, n={len(clean)})", xlabel="Predicted", ylabel="True")
    ax.grid(False)
    fig.colorbar(im, fraction=0.03)
    fig.savefig(FIG / "confusion_matrix_test.png")
    plt.close(fig)

    clean_acc = clean_metrics["accuracy"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for i, fam in enumerate([f for f in rob["family"].unique() if f != "clean"]):
        d = rob[rob["family"] == fam]
        ax[0].plot([0] + d["severity"].tolist(), [clean_acc] + d["accuracy"].tolist(), "o-", color=PALETTE[i], label=fam)
        ax[1].plot([0] + d["severity"].tolist(), [0] + d["prediction_change_rate"].tolist(), "o-", color=PALETTE[i], label=fam)
    ax[0].set(title="Test accuracy vs corruption severity", xlabel="Severity (0 = clean)", ylabel="Accuracy", xticks=[0, 1, 2, 3])
    ax[1].set(title="Prediction-change rate vs severity", xlabel="Severity (0 = clean)", ylabel="Share of predictions changed", xticks=[0, 1, 2, 3])
    ax[0].legend(fontsize=8)
    fig.savefig(FIG / "robustness_accuracy.png")
    plt.close(fig)
    print(json.dumps({k: v for k, v in clean_metrics.items() if k != "class_distribution"}, indent=2))
    print(rob.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
