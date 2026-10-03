import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from src import perturbations
from src.imaging import standardise
from src.quality import quality_features
from src.reliability import FEATURES, feature_vector, prob_features, stability_features
from src.reliability_metrics import best_threshold, binary_report


def test_prob_features_values():
    p = np.array([0.7, 0.2, 0.1])
    f = prob_features(p)
    assert f["max_prob"] == pytest.approx(0.7) and f["margin_top1_top2"] == pytest.approx(0.5)
    assert 0 < f["entropy_norm"] < 1 and f["top3_mass"] == pytest.approx(1.0)


def test_stability_features():
    p = np.array([0.6, 0.3, 0.1])
    pert = np.array([[0.5, 0.4, 0.1], [0.3, 0.6, 0.1]])
    f = stability_features(p, pert)
    assert f["stability_agree_frac"] == 0.5 and f["stability_n_distinct"] == 2
    assert f["stability_min_pred_prob"] == pytest.approx(0.3)


def test_quality_features_react_to_degradation(leaf_image):
    img = standardise(leaf_image)
    q = quality_features(img, leaf_image.size)
    assert q["original_min_side"] == 200
    blurred = quality_features(perturbations.apply(img, "gaussian_blur", 3), img.size)
    noisy = quality_features(perturbations.apply(img, "gaussian_noise", 0.2), img.size)
    dark = quality_features(perturbations.apply(img, "brightness_low", 0.3), img.size)
    assert blurred["sharpness_log_lapvar"] < q["sharpness_log_lapvar"]
    assert noisy["noise_sigma"] > q["noise_sigma"]
    assert dark["brightness"] < q["brightness"]


def test_feature_order_is_fixed_and_complete():
    feats = {name: float(i) for i, name in enumerate(FEATURES["B"])}
    v = feature_vector(feats, "A")
    assert v.shape == (1, len(FEATURES["A"])) and list(v.columns) == FEATURES["A"]
    assert list(v.iloc[0]) == [float(FEATURES["B"].index(n)) for n in FEATURES["A"]]
    assert FEATURES["B"][:len(FEATURES["A"])] == FEATURES["A"]


def test_reliability_prediction_and_threshold_selection():
    rng = np.random.default_rng(0)
    X = rng.random((400, len(FEATURES["A"])))
    y = (X[:, 0] < 0.3).astype(int)            # "low confidence -> incorrect"
    rf = RandomForestClassifier(n_estimators=50, random_state=0).fit(X, y)
    score = rf.predict_proba(X)[:, 1]
    t = best_threshold(y, score)
    rep = binary_report(y, score, t)
    assert 0 <= t <= 1 and rep["recall"] > 0.9 and rep["tp"] + rep["fn"] == y.sum()


def test_engine_features_match_app_predictor(tmp_path, leaf_image):
    """Training-time features (engine) must equal inference-time features (Predictor) for the same image."""
    import pandas as pd
    from src import engine, perturbations
    from src.config import load_config
    from src.model import build_model, save_classifier
    from src.predictor import Predictor
    cfg = load_config()
    names = ["Apple___healthy", "Tomato___healthy", "Corn_(maize)___healthy"]
    save_classifier(build_model(len(names), pretrained=False).eval(), names, tmp_path / "models")
    img_dir = tmp_path / "imgs" / names[0]
    img_dir.mkdir(parents=True)
    standardise_size = cfg["image"]["standard_size"]
    from src.imaging import standardise
    standardise(leaf_image, standardise_size).save(img_dir / "leaf.png")
    items = pd.DataFrame({"path": [f"{names[0]}/leaf.png"], "class_name": [names[0]], "image_id": [7]})
    pred = Predictor(models_dir=tmp_path / "models", device="cpu")
    rows = engine.run(pred.model, "cpu", cfg, tmp_path / "imgs", items, {n: i for i, n in enumerate(names)},
                      perturbations.conditions(cfg), all_conditions=False, workers=0)
    clean = rows[rows["family"] == "clean"].iloc[0]
    out = pred.analyse(standardise(leaf_image, standardise_size), with_stability=True)
    for name in FEATURES["B"]:
        if name in ("original_min_side", "aspect_ratio"):
            continue        # app measures the raw upload size; engine measures the condition image
        assert clean[name] == pytest.approx(out["features"][name], rel=1e-4, abs=1e-5), name
    assert set(rows["family"]) <= {c[0] for c in perturbations.conditions(cfg)} and len(rows) == 4
