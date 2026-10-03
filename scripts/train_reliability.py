"""Step 5: train the Random Forest reliability models on CALIBRATION data only.
rel_train rows fit the forests; rel_val rows choose hyper-parameters and every threshold. The test set is not read.

    python -m scripts.train_reliability
"""
import itertools
import json

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score

from src.config import load_config, project_path
from src.plots import GREEN, plt
from src.reliability import FEATURES
from src.reliability_metrics import best_threshold, binary_report

MET, FIG = project_path("reports/metrics"), project_path("reports/figures")


def baseline_scores(df: pd.DataFrame) -> dict:
    return {"confidence_threshold": 1 - df["max_prob"].to_numpy(),         # low confidence -> risky
            "margin_threshold": 1 - df["margin_top1_top2"].to_numpy()}     # small top-1/top-2 gap -> risky


def main():
    cfg = load_config()
    models_dir = project_path(cfg["paths"]["models_dir"])
    rows = pd.read_csv(project_path("reports/features/calibration_rows.csv.gz"))
    rows = rows[rows["has_reliability_features"] == 1]
    tr, va = rows[rows["reliability_role"] == "rel_train"], rows[rows["reliability_role"] == "rel_val"]
    assert not set(tr["image_id"]) & set(va["image_id"]), "an image appears in both rel_train and rel_val"
    ytr, yva = tr["incorrect"].to_numpy(), va["incorrect"].to_numpy()
    print(f"rel_train {len(tr)} rows ({ytr.mean():.3%} incorrect), rel_val {len(va)} rows ({yva.mean():.3%} incorrect)")

    # reference ranges of image-quality indicators on clean rel_train images (used only for the app's explanations)
    clean_tr = tr[tr["family"] == "clean"]
    from src.reliability import QUALITY_FEATURES
    (models_dir / "quality_reference.json").write_text(json.dumps(
        {k: {"p05": float(clean_tr[k].quantile(0.05)), "p95": float(clean_tr[k].quantile(0.95))} for k in QUALITY_FEATURES}, indent=2))

    grid = cfg["reliability"]["rf_search"]
    schema = {"positive_class": "1 = disease prediction is INCORRECT", "threshold_rule": "maximise F2 on rel_val",
              "stability_policy": cfg["stability_policy"], "sklearn_version": sklearn.__version__,
              "trained_on": "calibration split, rel_train rows (clean + 18 corruptions per image)", "configs": {}}
    selection = {}
    for name, feats in FEATURES.items():
        results = []
        for vals in itertools.product(*grid.values()):
            params = dict(zip(grid.keys(), vals))
            rf = RandomForestClassifier(**params, class_weight="balanced_subsample", n_jobs=-1, random_state=cfg["seed"])
            rf.fit(tr[feats], ytr)
            ap = average_precision_score(yva, rf.predict_proba(va[feats])[:, 1])
            results.append({"params": params, "val_pr_auc": float(ap)})
            print(f"  RF-{name} {params} val PR-AUC {ap:.4f}", flush=True)
        best = max(results, key=lambda r: r["val_pr_auc"])
        rf = RandomForestClassifier(**best["params"], class_weight="balanced_subsample", n_jobs=-1, random_state=cfg["seed"])
        rf.fit(tr[feats], ytr)
        score_va = rf.predict_proba(va[feats])[:, 1]
        thr = best_threshold(yva, score_va)
        joblib.dump(rf, models_dir / f"reliability_rf_{name}.joblib", compress=3)
        schema["configs"][name] = {"features": feats, "model_file": f"reliability_rf_{name}.joblib", "threshold": thr,
                                   "hyperparameters": best["params"],
                                   "extra_forward_passes": 0 if name == "A" else len(cfg["stability_policy"])}
        selection[f"RF-{name}"] = {"search": results, "selected": best, "rel_val": binary_report(yva, score_va, thr)}
        imp = permutation_importance(rf, va[feats], yva, scoring="average_precision", n_repeats=5,
                                     random_state=cfg["seed"], n_jobs=-1)
        pd.DataFrame({"feature": feats, "impurity_importance": rf.feature_importances_,
                      "permutation_importance_pr_auc": imp.importances_mean,
                      "permutation_std": imp.importances_std}).sort_values(
            "permutation_importance_pr_auc", ascending=False).to_csv(MET / f"reliability_feature_importance_{name}.csv", index=False)

    for name, s in baseline_scores(va).items():
        thr = best_threshold(yva, s)
        schema["configs_baselines"] = schema.get("configs_baselines", {})
        schema["configs_baselines"][name] = {"score": "1 - max_prob" if "confidence" in name else "1 - (top1 - top2)",
                                             "threshold_on_score": thr}
        selection[name] = {"rel_val": binary_report(yva, s, thr)}

    (models_dir / "reliability_schema.json").write_text(json.dumps(schema, indent=2))
    (models_dir / "reliability_threshold.json").write_text(json.dumps(
        {k: v["threshold"] for k, v in schema["configs"].items()}, indent=2))
    (MET / "reliability_model_selection.json").write_text(json.dumps(selection, indent=2))

    imp = pd.read_csv(MET / "reliability_feature_importance_B.csv")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(imp["feature"][::-1], imp["permutation_importance_pr_auc"][::-1],
            xerr=imp["permutation_std"][::-1], color=GREEN)
    ax.set(title="Reliability RF (config B): permutation importance on rel_val",
           xlabel="Drop in PR-AUC when the feature is shuffled")
    fig.savefig(FIG / "reliability_feature_importance.png")
    for k, v in selection.items():
        r = v["rel_val"]
        print(f"{k:22s} rel_val: P {r['precision']:.3f} R {r['recall']:.3f} F1 {r['f1']:.3f} PR-AUC {r['pr_auc']:.3f}")


if __name__ == "__main__":
    main()
