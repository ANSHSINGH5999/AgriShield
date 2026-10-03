"""Need the trained files in models/ (shipped with the project)."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HAS_MODELS = (ROOT / "models" / "efficientnet_b0.pt").exists() and (ROOT / "models" / "reliability_schema.json").exists()
needs_models = pytest.mark.skipif(not HAS_MODELS, reason="trained models not present")


def test_missing_model_file_gives_clear_error(tmp_path):
    from src.model import ModelFileError, load_classifier
    with pytest.raises(ModelFileError, match="Missing model file"):
        load_classifier(tmp_path)


@needs_models
def test_models_load_and_class_mapping_consistent():
    import json
    from src.model import load_classifier
    model, names = load_classifier(ROOT / "models", "cpu")
    assert model.classifier[1].out_features == len(names) == len(json.loads((ROOT / "models/class_names.json").read_text()))


@needs_models
def test_sample_image_gives_real_prediction_and_risk():
    from src.imaging import load_rgb
    from src.predictor import Predictor
    sample = sorted((ROOT / "assets" / "samples").glob("*.jpg"))[0]
    out = Predictor(device="cpu").analyse(load_rgb(sample))
    assert 0 <= out["confidence"] <= 1 and len(out["top3"]) == 3
    assert abs(sum(out["probs"]) - 1) < 1e-4
    assert 0 <= out["risk"] <= 1 and 0 <= out["stability"] <= 1


@needs_models
def test_app_starts_and_pages_render():
    from streamlit.testing.v1 import AppTest
    for page in ["Dashboard", "Disease Detection", "Robustness Lab", "Reliability Report", "Model Evaluation", "About"]:
        at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=240).run()
        at.sidebar.radio[0].set_value(page).run()
        assert not at.exception, f"{page}: {at.exception}"
