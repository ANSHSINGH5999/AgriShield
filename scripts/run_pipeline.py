"""Full training + evaluation pipeline, in the required order (test set used only in the last steps).

    python -m scripts.run_pipeline              (everything, ~2-4 h on a laptop GPU / Apple M-series; much longer on CPU)
    python -m scripts.run_pipeline --evaluate   (only the evaluation steps, using the saved models)
"""
import argparse
import subprocess
import sys

TRAIN = [["scripts.prepare_data"], ["scripts.train_classifier"], ["scripts.compute_features", "--split", "calibration"],
         ["scripts.train_reliability"]]
EVALUATE = [["scripts.compute_features", "--split", "test"], ["scripts.evaluate_classifier"], ["scripts.evaluate_reliability"],
            ["scripts.evaluate_external"], ["scripts.make_report"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evaluate", action="store_true", help="run only the evaluation steps")
    ap.add_argument("--train", action="store_true", help="run only the training steps")
    a = ap.parse_args()
    steps = EVALUATE if a.evaluate else TRAIN if a.train else TRAIN + EVALUATE
    for step in steps:
        print(f"\n=== python -m {' '.join(step)} ===", flush=True)
        if subprocess.call([sys.executable, "-m", *step]) != 0:
            sys.exit(f"Step failed: {' '.join(step)}")


if __name__ == "__main__":
    main()
