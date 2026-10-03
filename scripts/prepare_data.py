"""Step 1: inspect PlantVillage + PlantDoc, find duplicates, build leakage-safe splits and the label mapping.

    python -m scripts.prepare_data
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_config, project_path
from src.data import assign_splits, duplicate_groups, hash_images, list_images
from src.external_mapping import PLANTDOC_TO_PLANTVILLAGE

MAN = project_path("reports/manifests")
MET = project_path("reports/metrics")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--plantdoc-only", action="store_true", help="only (re)build the PlantDoc manifest")
    args = ap.parse_args()
    cfg = load_config()
    pv_root = project_path(cfg["paths"]["plantvillage_dir"])
    pd_root = project_path(cfg["paths"]["plantdoc_dir"])
    MAN.mkdir(parents=True, exist_ok=True)
    MET.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    if args.plantdoc_only:
        report = json.loads((MET / "dataset_report.json").read_text())
        usable = pd.read_csv(MAN / "plantvillage_splits.csv")
        cls_counts = usable["class_name"].value_counts()
        report["plantdoc"] = plantdoc_section(pd_root, usable, cls_counts)
        (MET / "dataset_report.json").write_text(json.dumps(report, indent=2))
        print(json.dumps({k: v for k, v in report["plantdoc"].items() if k != "class_counts"}, indent=2))
        return

    # ---------- PlantVillage ----------
    pv = list_images(pv_root)
    print(f"PlantVillage: {len(pv)} files in {pv['class_name'].nunique()} class folders; hashing ...", flush=True)
    pv, thumbs = hash_images(pv_root, pv)
    bad = pv[pv["read_error"] != ""]
    ok_mask = (pv["read_error"] == "").to_numpy()
    pv_ok = pv[ok_mask].reset_index(drop=True)
    pv_ok["group_id"], dup_stats = duplicate_groups(pv_ok, thumbs[ok_mask])
    mixed = pv_ok.groupby("group_id")["class_name"].nunique()
    mixed_groups = mixed[mixed > 1].index
    conflicts = pv_ok[pv_ok["group_id"].isin(mixed_groups)]
    # images whose (near-)duplicates carry a DIFFERENT label are ambiguous -> excluded from all splits
    usable = pv_ok[~pv_ok["group_id"].isin(mixed_groups)].reset_index(drop=True)
    usable["split"] = assign_splits(usable, cfg["split"], cfg["seed"])
    # reliability sub-split of the calibration partition (also by group)
    calib = usable[usable["split"] == "calibration"]
    rng = np.random.default_rng(cfg["seed"] + 1)
    groups = calib["group_id"].unique()
    val_groups = set(rng.choice(groups, int(round(len(groups) * cfg["split"]["reliability_val_fraction"])), replace=False))
    usable.loc[calib.index, "reliability_role"] = np.where(calib["group_id"].isin(val_groups), "rel_val", "rel_train")
    usable["image_id"] = np.arange(len(usable))
    usable.to_csv(MAN / "plantvillage_splits.csv", index=False)
    conflicts.to_csv(MAN / "plantvillage_excluded_label_conflicts.csv", index=False)
    bad.to_csv(MAN / "plantvillage_unreadable.csv", index=False)

    # leakage checks
    per_group = usable.groupby("group_id")["split"].nunique()
    assert (per_group == 1).all(), "a duplicate group spans several splits"
    for col in ("md5",):
        s = usable.groupby(col)["split"].nunique()
        assert (s == 1).all(), f"identical files ({col}) in different splits"

    counts = usable.pivot_table(index="class_name", columns="split", values="path", aggfunc="count", fill_value=0)
    counts = counts[["train", "val", "calibration", "test"]]
    counts.to_csv(MET / "plantvillage_class_split_counts.csv")
    cls_counts = pv_ok["class_name"].value_counts()
    sizes = pv_ok.groupby(["width", "height"]).size().sort_values(ascending=False)

    pd_ready = pd_root.exists() and any(d.is_dir() and d.name.lower() in ("train", "test") for d in pd_root.iterdir())
    pdoc_info = plantdoc_section(pd_root, usable, cls_counts) if pd_ready else {
        "status": "not downloaded yet - run: python -m scripts.prepare_data --plantdoc-only"}

    report = {
        "plantvillage": {
            "source": "https://github.com/spMohanty/PlantVillage-Dataset (raw/color)",
            "files": int(len(pv)), "unreadable": int(len(bad)), "classes": int(pv_ok["class_name"].nunique()),
            "class_counts": cls_counts.to_dict(),
            "imbalance_ratio_max_over_min": float(cls_counts.max() / cls_counts.min()),
            "image_sizes_top": {f"{w}x{h}": int(n) for (w, h), n in sizes.head(5).items()},
            "duplicates": {**dup_stats, "duplicate_groups_with_2plus_images": int((usable.groupby('group_id').size() > 1).sum()),
                           "images_in_label_conflict_groups_excluded": int(len(conflicts))},
            "split_method": "group-level stratified split by class; groups = exact copies (md5), confirmed near-duplicates "
                            "(dHash <= 4 bits AND thumbnail MAE <= 8) and photos of the same leaf (same class + source id)",
            "split_seed": cfg["seed"], "split_fractions": {k: cfg["split"][k] for k in ("train", "val", "calibration", "test")},
            "split_sizes": usable["split"].value_counts().to_dict(),
            "reliability_roles": usable["reliability_role"].value_counts().to_dict(),
        },
        "plantdoc": pdoc_info,
        "seconds": time.time() - t0,
    }
    (MET / "dataset_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk not in ("class_counts",)} if isinstance(v, dict) else v
                      for k, v in report.items()}, indent=2))


def plantdoc_section(pd_root, usable, cls_counts) -> dict:
    pdoc_rows = []
    for part in ("train", "test"):
        part_dir = next((d for d in pd_root.iterdir() if d.is_dir() and d.name.lower() == part), None)
        if part_dir is None:
            continue
        df = list_images(part_dir)
        df["path"] = part_dir.name + "/" + df["path"]
        df["plantdoc_split"] = part
        pdoc_rows.append(df)
    pdoc = pd.concat(pdoc_rows, ignore_index=True)
    pdoc, _ = hash_images(pd_root, pdoc)
    pdoc["mapped_class"] = pdoc["class_name"].map(lambda c: PLANTDOC_TO_PLANTVILLAGE.get(c, {}).get("plantvillage"))
    pdoc["mapping_status"] = pdoc["class_name"].map(lambda c: PLANTDOC_TO_PLANTVILLAGE.get(c, {}).get("status", "unmapped"))
    # an external image identical to a PlantVillage training/val image would not be independent
    pv_md5 = set(usable["md5"])
    pdoc["md5_also_in_plantvillage"] = pdoc["md5"].isin(pv_md5)
    pdoc.to_csv(MAN / "plantdoc_manifest.csv", index=False)
    missing_targets = sorted({v["plantvillage"] for v in PLANTDOC_TO_PLANTVILLAGE.values()
                              if v.get("plantvillage") and v["plantvillage"] not in set(cls_counts.index)})

    return {
        "source": "https://github.com/pratikkayal/PlantDoc-Dataset (TRAIN + TEST folders; never trained on)",
        "files": int(len(pdoc)), "unreadable": int((pdoc["read_error"] != "").sum()),
        "classes": int(pdoc["class_name"].nunique()), "class_counts": pdoc["class_name"].value_counts().to_dict(),
        "unmapped_classes": sorted(set(pdoc.loc[pdoc["mapping_status"] == "unmapped", "class_name"])),
        "mapping_status_counts": pdoc["mapping_status"].value_counts().to_dict(),
        "identical_to_plantvillage_files": int(pdoc["md5_also_in_plantvillage"].sum()),
        "mapping_targets_missing_in_plantvillage": missing_targets,
    }


if __name__ == "__main__":
    main()
