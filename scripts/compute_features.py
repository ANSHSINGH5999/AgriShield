"""Step 3: run the trained classifier on every (image, condition) of a split and compute reliability features.

    python -m scripts.compute_features --split calibration
    python -m scripts.compute_features --split test        (the locked test set - run once, after everything is final)
"""
import argparse
import json
import time

import pandas as pd

from src import perturbations
from src.config import get_device, load_config, project_path, set_seed
from src.engine import run
from src.model import load_classifier


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["calibration", "test"], required=True)
    ap.add_argument("--limit", type=int, help="first N images only (smoke test)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--batch-images", type=int, default=3, help="images per GPU batch (lower it if memory is tight)")
    args = ap.parse_args()
    cfg = load_config()
    set_seed(cfg["seed"])
    device = get_device()
    model, class_names = load_classifier(project_path(cfg["paths"]["models_dir"]), device)
    idx = {c: i for i, c in enumerate(class_names)}
    man = pd.read_csv(project_path("reports/manifests/plantvillage_splits.csv"))
    items = man[man["split"] == args.split].reset_index(drop=True)
    if args.limit:
        items = items.head(args.limit)
    conds = perturbations.conditions(cfg)
    print(f"{args.split}: {len(items)} images on {device}", flush=True)
    t0 = time.time()
    # test: every condition for the robustness study (+ reliability features on clean + 3 sampled conditions);
    # calibration: only the reliability conditions are needed
    rows = run(model, device, cfg, project_path(cfg["paths"]["plantvillage_dir"]), items, idx, conds,
               all_conditions=args.split == "test", batch_images=args.batch_images, workers=args.workers)
    if args.split == "calibration":
        rows = rows.merge(items[["image_id", "reliability_role"]], on="image_id")
    out = project_path("reports/features")
    out.mkdir(parents=True, exist_ok=True)
    suffix = "_smoke" if args.limit else ""
    rows.to_csv(out / f"{args.split}_rows{suffix}.csv.gz", index=False)
    (out / f"{args.split}_run{suffix}.json").write_text(json.dumps({
        "images": len(items), "conditions": len(conds), "rows": len(rows), "seconds": time.time() - t0, "device": device}, indent=2))
    print(f"saved {len(rows)} rows in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
