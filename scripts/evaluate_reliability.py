"""Step 6: evaluate the FROZEN reliability models and baselines on the locked TEST rows (thresholds from rel_val).

    python -m scripts.evaluate_reliability
"""
import json

import joblib
import numpy as np
import pandas as pd

from src.config import load_config, project_path
from src.plots import PALETTE, plt
from src.reliability_metrics import binary_report, calibration_table, risk_coverage

MET, FIG = project_path("reports/metrics"), project_path("reports/figures")


def method_scores(df, models_dir, schema):
    out = {"Confidence threshold": (1 - df["max_prob"].to_numpy(), schema["configs_baselines"]["confidence_threshold"]["threshold_on_score"]),
           "Margin threshold": (1 - df["margin_top1_top2"].to_numpy(), schema["configs_baselines"]["margin_threshold"]["threshold_on_score"])}
    for name, info in schema["configs"].items():
        rf = joblib.load(models_dir / info["model_file"])
        out[f"Random Forest ({name})"] = (rf.predict_proba(df[info["features"]])[:, 1], info["threshold"])
    return out


def main():
    cfg = load_config()
    models_dir = project_path(cfg["paths"]["models_dir"])
    schema = json.loads((models_dir / "reliability_schema.json").read_text())
    rows = pd.read_csv(project_path("reports/features/test_rows.csv.gz"))
    rows = rows[rows["has_reliability_features"] == 1].reset_index(drop=True)   # clean + 3 sampled corruptions per image
    y = rows["incorrect"].to_numpy()
    scores = method_scores(rows, models_dir, schema)
    results, subsets = {}, {"all conditions": np.ones(len(rows), bool), "clean only": (rows["family"] == "clean").to_numpy(),
                             "corrupted only": (rows["family"] != "clean").to_numpy()}
    for name, (s, thr) in scores.items():
        results[name] = {sub: binary_report(y[m], s[m], thr) for sub, m in subsets.items()}
        cov, err, aurc = risk_coverage(y, s)
        results[name]["all conditions"]["aurc"] = aurc
        results[name]["risk_coverage"] = {"coverage": cov.tolist(), "error_rate": err.tolist()}
    cal, brier = calibration_table(y, scores["Random Forest (B)"][0])
    cal_a, brier_a = calibration_table(y, scores["Random Forest (A)"][0])
    out = {"positive_class": "incorrect disease prediction", "test_rows": int(len(rows)),
           "test_images": int(rows["image_id"].nunique()), "conditions_per_image": int(rows.groupby("image_id").size().iloc[0]),
           "condition_sampling": "clean + 3 corruption conditions sampled uniformly (seeded per image) from the 18",
           "methods": results, "calibration_B": {"bins": cal, "brier": brier}, "calibration_A": {"bins": cal_a, "brier": brier_a}}
    (MET / "reliability_test_metrics.json").write_text(json.dumps(out, indent=2))

    table = pd.DataFrame([{"method": n, **{k: r["all conditions"][k] for k in
                          ("precision", "recall", "f1", "roc_auc", "pr_auc", "false_negative_rate", "false_alarm_rate", "flagged_share", "aurc")}}
                          for n, r in results.items() if n != "risk_coverage"])
    table.to_csv(MET / "reliability_baseline_comparison.csv", index=False)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
    for i, name in enumerate(scores):
        rc = results[name]["risk_coverage"]
        ax[0].plot(rc["coverage"], rc["error_rate"], color=PALETTE[i], label=f"{name} (AURC {results[name]['all conditions']['aurc']:.4f})")
    ax[0].set(title="Risk vs coverage (test, all conditions)", xlabel="Share of predictions accepted", ylabel="Error rate among accepted")
    ax[0].legend(fontsize=8)
    c = pd.DataFrame(cal)
    ax[1].plot([0, 1], [0, 1], "--", color="#999", label="perfect calibration")
    ax[1].plot(c["mean_predicted"], c["observed_incorrect"], "o-", color=PALETTE[0], label=f"RF (B), Brier {brier:.4f}")
    ax[1].set(title="Reliability calibration (test)", xlabel="Predicted P(incorrect)", ylabel="Observed share incorrect")
    ax[1].legend(fontsize=8)
    fig.savefig(FIG / "reliability_risk_coverage_calibration.png")
    print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
