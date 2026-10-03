"""Need the trained files in models/ (shipped with the project)."""
import re
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


PAGES = ["Disease Detection", "Model Performance", "About"]


def _app():
    from streamlit.testing.v1 import AppTest
    return AppTest.from_file(str(ROOT / "app.py"), default_timeout=300)


def _all_text(at) -> str:
    parts = [m.value for m in at.markdown if not m.value.lstrip().startswith("<style>")] + \
            [c.value for c in at.caption] + [h.value for h in at.header] + \
            [i.value for i in at.info] + [str(df.value) for df in at.dataframe]
    return "\n".join(str(p) for p in parts)


@needs_models
def test_navigation_has_only_three_pages_and_they_render():
    at = _app().run()
    assert list(at.sidebar.radio[0].options) == PAGES
    for page in PAGES:
        at = _app().run()
        at.sidebar.radio[0].set_value(page).run()
        assert not at.exception, f"{page}: {at.exception}"


@needs_models
def test_one_upload_runs_the_whole_demo_automatically():
    at = _app().run()
    at.selectbox[0].set_value(sorted(p.name for p in (ROOT / "assets" / "samples").glob("*.jpg"))[0]).run()
    assert not at.exception
    text = _all_text(at)
    for heading in ["Disease prediction", "Reliability assessment", "Automatic robustness test", "Final assessment"]:
        assert heading in text, heading
    assert "Estimated error risk" in text and "Reference" in text
    assert re.search(r"Prediction agreement: [0-5]/5 degraded conditions", text)
    assert ("Prediction appears reliable" in text) != ("may be unreliable under changed image conditions" in text)
    # clean main page: no manual controls, no internal threshold or feature details
    assert len(at.slider) == 0 and len(at.toggle) == 0
    for internal in ["Warning at", "warning at", "threshold", "entropy", "margin", ".joblib", "PlantDoc"]:
        assert internal not in text, internal


@needs_models
def test_plantdoc_only_on_model_performance_and_clearly_labelled():
    at = _app().run()
    at.sidebar.radio[0].set_value("Model Performance").run()
    assert not at.exception
    assert "External PlantDoc Evaluation" in _all_text(at)
    at = _app().run()
    assert "PlantDoc" not in _all_text(at)            # default page = Disease Detection


def test_no_novelty_claims_in_the_app():
    text = (ROOT / "app.py").read_text().lower()
    for claim in ["novel", "first-ever", "world's first", "state-of-the-art"]:
        assert claim not in text, claim


def test_robustness_demo_uses_existing_config_strengths():
    from src.config import load_config
    from src.robustness_demo import DEMO_SEVERITY, demo_conditions
    cfg = load_config()
    for _, fam, val in demo_conditions(cfg):
        assert val == float(cfg["perturbations"][fam][DEMO_SEVERITY - 1])
