"""Step 7: external evaluation on PlantDoc (never used for training). Uses the same Predictor as the app.

    python -m scripts.evaluate_external
"""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from src.config import load_config, project_path
from src.imaging import load_rgb
from src.predictor import Predictor
from src.reliability_metrics import binary_report

MET = project_path("reports/metrics")


def main():
    cfg = load_config()
    root = project_path(cfg["paths"]["plantdoc_dir"])
    man = pd.read_csv(project_path("reports/manifests/plantdoc_manifest.csv"))
    eligible = man[(man["mapping_status"].isin(["exact", "ambiguous"])) & (man["read_error"].isna() | (man["read_error"] == ""))
                   & (~man["md5_also_in_plantvillage"])].reset_index(drop=True)
    pred = Predictor()
    idx = {c: i for i, c in enumerate(pred.class_names)}
    rows, failures = [], []
    for i, r in eligible.iterrows():
        try:
            out = pred.analyse(load_rgb(root / r["path"]), with_stability=True)
        except Exception as e:                      # recorded, not hidden
            failures.append({"path": r["path"], "error": str(e)})
            continue
        true = idx[r["mapped_class"]]
        rows.append({"path": r["path"], "plantdoc_class": r["class_name"], "mapped_class": r["mapped_class"],
                     "mapping_status": r["mapping_status"], "true_index": true, "pred_index": out["pred_index"],
                     "confidence": out["confidence"], "correct": int(out["pred_index"] == true),
                     "top3_correct": int(true in [idx[c] for c, _ in out["top3"]]),
                     "risk_B": out.get("risk"), "stability": out.get("stability"),
                     "risk_A": pred.reliability.risk(out["features"], "A") if pred.reliability else None})
        if (i + 1) % 250 == 0:
            print(f"  {i + 1}/{len(eligible)}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(MET / "external_plantdoc_predictions.csv", index=False)

    def summary(d):
        labels = sorted(d["true_index"].unique())
        y = d["incorrect"] = 1 - d["correct"]
        res = {"images": int(len(d)), "classes": int(len(labels)), "accuracy": float(accuracy_score(d["true_index"], d["pred_index"])),
               "top3_accuracy": float(d["top3_correct"].mean()),
               "macro_f1_over_present_classes": float(f1_score(d["true_index"], d["pred_index"], labels=labels, average="macro", zero_division=0)),
               "mean_confidence": float(d["confidence"].mean())}
        if pred.reliability and 0 < y.mean() < 1:
            for cfg_name in ("A", "B"):
                res[f"reliability_{cfg_name}"] = binary_report(y.to_numpy(), d[f"risk_{cfg_name}"].to_numpy(), pred.reliability.threshold(cfg_name))
        return res

    strict = df[df["mapping_status"] == "exact"].copy()
    out = {"dataset": "PlantDoc (TRAIN + TEST folders, external only)",
           "eligible_images": int(len(eligible)), "evaluated_images": int(len(df)), "failures": failures,
           "excluded": {"unmapped_or_no_match": int((~man["mapping_status"].isin(["exact", "ambiguous"])).sum()),
                        "unreadable": int((man["read_error"].fillna("") != "").sum()),
                        "identical_to_plantvillage": int(man["md5_also_in_plantvillage"].sum())},
           "strict_exact_mappings": summary(strict), "extended_with_ambiguous": summary(df.copy()),
           "per_class_accuracy_strict": strict.groupby("plantdoc_class")["correct"].agg(["mean", "size"]).rename(
               columns={"mean": "accuracy", "size": "images"}).reset_index().to_dict(orient="records"),
           "note": "PlantDoc images are field photos (cluttered backgrounds, several leaves); PlantVillage images are single "
                   "leaves on plain backgrounds. Results are reported separately and are not combined with PlantVillage results."}
    (MET / "external_plantdoc_metrics.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ("eligible_images", "evaluated_images", "excluded")}, indent=2))
    for k in ("strict_exact_mappings", "extended_with_ambiguous"):
        s = out[k]
        print(k, {kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in s.items() if not kk.startswith("reliability")})


if __name__ == "__main__":
    main()
