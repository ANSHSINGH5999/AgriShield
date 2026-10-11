"""One command to produce every evaluation graph + metric file from the saved models/results (no retraining).

    python generate_results.py

Reads:  models/ (RF reliability models + schema), reports/metrics/*, reports/features/test_rows.csv.gz
Writes: results/graphs/*.png, results/metrics.json, results/metrics.csv, results/model_comparison.{csv,png}
"""
import json
import textwrap

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from src.config import load_config, project_path
from src.plots import GREEN, PALETTE, plt
from src.reliability_metrics import binary_report, calibration_table

RES = project_path("results")
GR = RES / "graphs"
MET = project_path("reports/metrics")
GR.mkdir(parents=True, exist_ok=True)

FAMILY_LABEL = {"brightness_low": "Brightness (dark)", "brightness_high": "Brightness (bright)", "gaussian_blur": "Blur",
                "gaussian_noise": "Noise", "jpeg_compression": "JPEG", "low_resolution": "Low resolution"}
generated, skipped = [], []


def save(fig, name):
    fig.savefig(GR / name)
    plt.close(fig)
    generated.append(name)


def not_generated(name, reason, needed):
    skipped.append(name)
    print(f"[NOT GENERATED]\nGraph name: {name}\nReason: {reason}\nRequired data: {needed}\n")


def main():
    cfg = load_config()
    clf = json.loads((MET / "classifier_test_metrics.json").read_text())
    ext = json.loads((MET / "external_plantdoc_metrics.json").read_text())
    rob = pd.read_csv(MET / "robustness_by_corruption.csv")
    cm = pd.read_csv(MET / "test_confusion_matrix.csv", index_col=0)
    models_dir = project_path(cfg["paths"]["models_dir"])
    schema = json.loads((models_dir / "reliability_schema.json").read_text())

    # ---- Graph 1: EfficientNet-B0 performance
    names = ["Accuracy", "Precision (macro)", "Recall (macro)", "F1 (macro)"]
    vals = [clf["accuracy"], clf["macro_precision"], clf["macro_recall"], clf["macro_f1"]]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    bars = ax.bar(names, [v * 100 for v in vals], color=GREEN)
    ax.bar_label(bars, labels=[f"{v * 100:.2f}%" for v in vals], padding=3)
    ax.set(ylim=(95, 100.8), ylabel="%", title=f"EfficientNet-B0 on PlantVillage test set ({clf['test_images']:,} images)")
    save(fig, "efficientnet_performance.png")

    # ---- Graph 2: confusion matrix (row-normalised so 38 classes stay readable)
    mat = cm.to_numpy(float)
    norm = mat / mat.sum(1, keepdims=True)
    short = [c.replace("___", " / ").replace("_", " ") for c in cm.index]
    fig, ax = plt.subplots(figsize=(15, 13))
    im = ax.imshow(norm, cmap="Greens", vmin=0, vmax=1)
    ax.grid(False)
    ax.set_xticks(range(len(short)), short, rotation=90, fontsize=7)
    ax.set_yticks(range(len(short)), short, fontsize=7)
    ax.set(xlabel="Predicted class", ylabel="Actual class",
           title=f"EfficientNet-B0 confusion matrix (row-normalised; {int(mat.sum()):,} test images, {int(mat.sum() - np.trace(mat))} misclassified)")
    for i, j in zip(*np.nonzero(mat - np.diag(np.diag(mat)))):  # annotate only the errors with their counts
        ax.text(j, i, int(mat[i, j]), ha="center", va="center", fontsize=8, color="#c0392b", fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.03, label="Share of actual class")
    save(fig, "confusion_matrix.png")

    # ---- Graph 3 & 4: degradation robustness
    clean_acc = float(rob.loc[rob["family"] == "clean", "accuracy"].iloc[0])
    deg = rob[rob["family"] != "clean"]
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (fam, d) in enumerate(deg.groupby("family")):
        d = d.sort_values("severity")
        ax.plot([0, *d["severity"]], [clean_acc * 100, *d["accuracy"] * 100], "o-", color=PALETTE[i % len(PALETTE)],
                label=FAMILY_LABEL.get(fam, fam))
    ax.set(xticks=[0, 1, 2, 3], xticklabels=["Clean", "1 (mild)", "2", "3 (severe)"], xlabel="Degradation strength",
           ylabel="Accuracy (%)", title="EfficientNet-B0 accuracy under image degradation")
    ax.legend(fontsize=8)
    save(fig, "degradation_robustness.png")

    worst = deg.loc[deg.groupby("family")["accuracy"].idxmin()].assign(drop=lambda d: (clean_acc - d["accuracy"]) * 100)
    worst = worst.sort_values("drop")
    labels = [f"{FAMILY_LABEL.get(f, f)}\n(severity {s})" for f, s in zip(worst["family"], worst["severity"])]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    bars = ax.barh(labels, worst["drop"], color=PALETTE[4])
    ax.bar_label(bars, labels=[f"-{v:.1f} pts" for v in worst["drop"]], padding=3)
    ax.set(xlabel="Accuracy drop vs clean (percentage points)", xlim=(0, worst["drop"].max() * 1.15),
           title=f"Accuracy drop at the most severe level (clean = {clean_acc * 100:.2f}%)")
    save(fig, "accuracy_drop.png")

    # ---- Reliability model: score every stored test row with the saved RF models
    rows = pd.read_csv(project_path("reports/features/test_rows.csv.gz"))
    rows = rows[rows["has_reliability_features"] == 1].reset_index(drop=True)
    y = rows["incorrect"].to_numpy()
    info = schema["configs"]["B"]
    rf = joblib.load(models_dir / info["model_file"])
    risk = rf.predict_proba(rows[info["features"]])[:, 1]
    conf_risk = 1 - rows["max_prob"].to_numpy()
    thr_conf = schema["configs_baselines"]["confidence_threshold"]["threshold_on_score"]
    rf_rep, conf_rep = binary_report(y, risk, info["threshold"]), binary_report(y, conf_risk, thr_conf)
    _, brier = calibration_table(y, risk)

    # ---- Graph 5: PR curve
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for s, lab, c, rep in ((risk, "Random Forest (B)", GREEN, rf_rep), (conf_risk, "Confidence baseline (1 - max prob)", PALETTE[4], conf_rep)):
        p, r, _ = precision_recall_curve(y, s)
        ax.plot(r, p, color=c, label=f"{lab}, PR-AUC {rep['pr_auc']:.3f}")
    ax.axhline(y.mean(), ls="--", color="#999", label=f"Random guess ({y.mean():.3f})")
    ax.set(xlabel="Recall (incorrect predictions caught)", ylabel="Precision", ylim=(0, 1.02),
           title=f"Reliability model: precision-recall ({len(y):,} test rows)")
    ax.legend(fontsize=8, loc="lower left")
    save(fig, "reliability_pr_curve.png")

    # ---- Graph 6: confidence vs correctness
    bins = np.array([0, .5, .8, .9, .95, .99, .999, 1.0001])
    b = pd.cut(rows["max_prob"], bins, right=False)
    g = pd.DataFrame({"b": b, "inc": y, "risk": risk}).groupby("b", observed=True).agg(n=("inc", "size"), err=("inc", "mean"), rf=("risk", "mean"))
    g = g[g["n"] >= 30]
    if len(g) >= 3:
        fig, ax = plt.subplots(figsize=(8, 4.8))
        xl = [f"{iv.left:g}-{min(iv.right, 1):g}\n(n={n:,})" for iv, n in zip(g.index, g["n"])]
        ax.bar(xl, g["err"] * 100, color=PALETTE[4], label="Actual error rate in this confidence band")
        ax.plot(xl, g["rf"] * 100, "o-", color=GREEN, label="Mean Random Forest risk")
        ax.set(xlabel="EfficientNet-B0 confidence (max softmax)", ylabel="% of predictions incorrect",
               title="Confidence vs actual correctness (bands with >=30 rows)")
        ax.legend(fontsize=8)
        save(fig, "confidence_vs_correctness.png")
    else:
        not_generated("confidence_vs_correctness.png", "fewer than 3 confidence bands with >=30 rows", "per-prediction confidence + correctness")

    # ---- Graph 7: risk distribution
    fig, ax = plt.subplots(figsize=(8, 4.8))
    edges = np.linspace(0, 1, 41)
    ax.hist(risk[y == 0], edges, alpha=.65, color=GREEN, label=f"Correct (n={int((y == 0).sum()):,})", density=True)
    ax.hist(risk[y == 1], edges, alpha=.65, color=PALETTE[4], label=f"Incorrect (n={int(y.sum()):,})", density=True)
    ax.axvline(info["threshold"], ls="--", color="k", label=f"Flag threshold {info['threshold']:.3f}")
    ax.set(xlabel="Random Forest risk score P(incorrect)", ylabel="Density", yscale="log",
           title="Reliability risk: correct vs incorrect EfficientNet predictions")
    ax.legend(fontsize=8)
    save(fig, "reliability_risk_distribution.png")

    # ---- Metrics files
    deg_tbl = rob.assign(accuracy_drop_pts=(clean_acc - rob["accuracy"]) * 100)
    ext_s = ext["strict_exact_mappings"]
    metrics = {
        "efficientnet_b0": {"test_images": clf["test_images"], "accuracy": clf["accuracy"], "macro_precision": clf["macro_precision"],
                            "macro_recall": clf["macro_recall"], "macro_f1": clf["macro_f1"]},
        "robustness": deg_tbl[["family", "severity", "value", "accuracy", "macro_f1", "accuracy_drop_pts"]].to_dict("records"),
        "random_forest_reliability": {"rows": len(y), "incorrect_rate": float(y.mean()), "pr_auc": rf_rep["pr_auc"], "roc_auc": rf_rep["roc_auc"],
                                      "recall": rf_rep["recall"], "precision": rf_rep["precision"], "brier": brier,
                                      "confidence_baseline_pr_auc": conf_rep["pr_auc"], "confidence_baseline_recall": conf_rep["recall"]},
        "plantdoc_external": {"images": ext_s["images"], "accuracy": ext_s["accuracy"], "macro_f1": ext_s["macro_f1_over_present_classes"]},
    }
    (RES / "metrics.json").write_text(json.dumps(metrics, indent=2))
    flat = [("EfficientNet-B0", k, v) for k, v in metrics["efficientnet_b0"].items()]
    flat += [("Random Forest", k, v) for k, v in metrics["random_forest_reliability"].items()]
    flat += [("PlantDoc external", k, v) for k, v in metrics["plantdoc_external"].items()]
    flat += [("Robustness", f"{r['family']} sev{r['severity']} accuracy", r["accuracy"]) for r in metrics["robustness"]]
    pd.DataFrame(flat, columns=["group", "metric", "value"]).to_csv(RES / "metrics.csv", index=False)

    # ---- Model comparison (different tasks -> different metrics)
    cmp_rows = [
        ("Model type", "Convolutional neural network (torchvision EfficientNet-B0, fine-tuned)", "Ensemble of decision trees (scikit-learn Random Forest)"),
        ("Purpose", "Classify the plant disease (38 classes)", "Predict whether EfficientNet's prediction is wrong"),
        ("Input", "Leaf image (RGB)", f"{len(info['features'])} numbers: confidence/entropy, image quality, prediction-stability features"),
        ("Output", "Disease class + confidence", "Risk score P(prediction incorrect) -> reliable / flag for review"),
        ("Role", "Primary predictor", "Safety layer on top of the predictor"),
        ("Main metric", "Accuracy, macro-F1 (clean test set)", "PR-AUC, recall of incorrect predictions, Brier score"),
        ("Actual result", f"Acc {clf['accuracy'] * 100:.2f}%, P {clf['macro_precision']:.3f}, R {clf['macro_recall']:.3f}, F1 {clf['macro_f1']:.3f}; "
                          f"PlantDoc external acc {ext_s['accuracy'] * 100:.1f}%",
         f"PR-AUC {rf_rep['pr_auc']:.3f} (confidence baseline {conf_rep['pr_auc']:.3f}), recall {rf_rep['recall'] * 100:.1f}%, "
         f"precision {rf_rep['precision'] * 100:.1f}%, Brier {brier:.3f}"),
        ("Strength", "Near-perfect on clean lab images", "Catches most wrong predictions, incl. under degradation; beats confidence alone"),
        ("Limitation", f"Accuracy collapses under severe blur/noise/JPEG/low-res; poor on field photos ({ext_s['accuracy'] * 100:.0f}%)",
         "Trained on PlantVillage-style errors; flags ~{:.0f}% of predictions (false alarms)".format(rf_rep["flagged_share"] * 100)),
    ]
    cdf = pd.DataFrame(cmp_rows, columns=["Parameter", "EfficientNet-B0", "Random Forest"])
    cdf.to_csv(RES / "model_comparison.csv", index=False)
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.axis("off")
    t = ax.table(cellText=[[textwrap.fill(str(c), 62) for c in r] for r in cdf.to_numpy()], colLabels=list(cdf.columns), loc="center", cellLoc="left",
                 colWidths=[0.12, 0.44, 0.44])
    t.auto_set_font_size(False)
    t.set_fontsize(7.5)
    t.scale(1, 3.4)
    for (r, c), cell in t.get_celld().items():
        cell.set_text_props(wrap=True)
        if r == 0:
            cell.set_facecolor(GREEN)
            cell.set_text_props(color="w", fontweight="bold")
    save(fig, "model_comparison.png")

    # ---- Terminal summary
    e, rb = metrics["efficientnet_b0"], rob.set_index(["family", "severity"])["accuracy"]
    bar = "=" * 40
    sub = "-" * 40
    print(f"\n{bar}\nAGRISHIELD MODEL EVALUATION\n{bar}\n\nDataset: PlantVillage (locked test set)\nTest samples: {e['test_images']:,}\n")
    print(f"{sub}\nEFFICIENTNET-B0 (macro-averaged over 38 classes)\n{sub}")
    for k in ("accuracy", "macro_precision", "macro_recall", "macro_f1"):
        print(f"{k:16s}: {e[k] * 100:.2f}%")
    print(f"\n{sub}\nROBUSTNESS (accuracy; most severe level | drop vs clean)\n{sub}\nClean            : {clean_acc * 100:.2f}%")
    for fam, d in deg.groupby("family"):
        w = d.loc[d["accuracy"].idxmin()]
        print(f"{FAMILY_LABEL.get(fam, fam):17s}: {w['accuracy'] * 100:.2f}%  (sev {int(w['severity'])}, drop {(clean_acc - w['accuracy']) * 100:.1f} pts)")
    print(f"\n{sub}\nRANDOM FOREST RELIABILITY MODEL ({len(y):,} test rows, {y.mean() * 100:.1f}% incorrect)\n{sub}")
    print(f"PR-AUC           : {rf_rep['pr_auc']:.3f}   (confidence baseline {conf_rep['pr_auc']:.3f})")
    print(f"Recall           : {rf_rep['recall'] * 100:.1f}%   (precision {rf_rep['precision'] * 100:.1f}%)")
    print(f"Brier score      : {brier:.3f}")
    print(f"\nPlantDoc external accuracy: {ext_s['accuracy'] * 100:.1f}% ({ext_s['images']:,} images)")
    print(f"\n{sub}\nGRAPHS GENERATED  ->  {GR}\n{sub}")
    for i, n in enumerate(generated, 1):
        print(f"{i}. {n}")
    print(f"\nAlso: results/metrics.json, metrics.csv, model_comparison.csv\n{bar}")


if __name__ == "__main__":
    main()
