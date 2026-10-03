"""AgriShield - plant disease detection with automatic reliability and robustness checks (research demonstration).

Run:  streamlit run app.py      (or double-click run_app.bat on Windows)
Upload ONE leaf photo; everything else runs automatically with the saved models. Nothing is retrained.
Every metric on "Model Performance" is read from reports/metrics/*.
"""
import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from src import robustness_demo  # noqa: E402
from src.imaging import ImageError, load_rgb, validate_upload  # noqa: E402
from src.model import ModelFileError  # noqa: E402
from src.predictor import Predictor, pretty  # noqa: E402
from src.ui import FOREST, LEAF, banner, big, card, hero, inject_css, pct, step  # noqa: E402

st.set_page_config(page_title="AgriShield", page_icon="🌿", layout="wide")
inject_css()
METRICS = ROOT / "reports" / "metrics"
FIGURES = ROOT / "reports" / "figures"
SAMPLES = ROOT / "assets" / "samples"


@st.cache_resource(show_spinner="Loading the trained models ...")
def get_predictor() -> Predictor:
    return Predictor()


@st.cache_data
def read_json(name: str):
    p = METRICS / name
    return json.loads(p.read_text()) if p.exists() else None


@st.cache_data
def read_csv(name: str):
    p = METRICS / name
    return pd.read_csv(p) if p.exists() else None


def need_setup(what: str) -> None:
    banner("warn", f"<b>{what} not available.</b> Run the evaluation steps (see README: <code>evaluate_windows.bat</code>). "
                   "No numbers are shown until they are measured.")


def image_input():
    """One upload (or a bundled sample). Returns (PIL image, name) or (None, None)."""
    samples = sorted(SAMPLES.glob("*.jpg")) + sorted(SAMPLES.glob("*.png"))
    up = st.file_uploader("Upload a plant leaf image (JPG, JPEG or PNG, up to 15 MB)", type=["jpg", "jpeg", "png"])
    choice = st.selectbox("No image at hand? Use a sample leaf instead", ["—"] + [s.name for s in samples])
    try:
        if up is not None:
            validate_upload(up.name, up.size)
            return load_rgb(up), up.name
        if choice != "—":
            return load_rgb(SAMPLES / choice), choice
    except ImageError as e:
        banner("bad", f"<b>Could not use this file.</b> {e}")
    return None, None


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("## 🌿 AgriShield")
    st.markdown("Plant disease detection & prediction reliability")
    page = st.radio("Navigate", ["Disease Detection", "Model Performance", "About"], label_visibility="collapsed")
    st.markdown('<div class="disclaimer">Research demonstration only - not a substitute for diagnosis by an agricultural expert.</div>',
                unsafe_allow_html=True)

# ---------------------------------------------------------------- Disease Detection (the whole demo)
if page == "Disease Detection":
    hero("AgriShield", "Plant Disease Detection & Prediction Reliability",
         "Upload one leaf photo. The disease prediction, the reliability check and the robustness test all run automatically.")
    try:
        predictor = get_predictor()
    except ModelFileError as e:
        banner("bad", f"<b>Model not loaded.</b> {e}")
        st.stop()
    except Exception as e:      # shown to the user instead of a stack trace
        banner("bad", f"<b>The model could not be loaded</b> ({e.__class__.__name__}: {e}).")
        st.stop()

    img, name = image_input()
    if img is None:
        st.info("Upload a leaf image to start. Everything after that is automatic.")
        st.stop()

    with st.spinner("Analysing the image: disease prediction, 6-copy stability check, reliability model, robustness test ..."):
        res = predictor.analyse(img, with_stability=True)        # EfficientNet + stability copies + Random Forest B
        rob = robustness_demo.run(predictor, img)                 # 5 representative degradations

    step(1, "🖼️ Uploaded image")
    c_img, _ = st.columns([2, 3])
    c_img.image(img, caption=f"{name} · {img.width}×{img.height} px", use_container_width=True)

    step(2, "🌿 Disease prediction")
    c = st.columns(3)
    with c[0]: big("Predicted disease", pretty(res["pred_class"]))
    with c[1]: big("Confidence", pct(res["confidence"]), "the model's probability for this class on this image (not accuracy)")
    with c[2]: big("Classifier", "EfficientNet-B0", "fine-tuned on PlantVillage, 38 classes")

    step(3, "🛡️ Prediction reliability")
    if "risk" not in res:
        need_setup("The reliability model")
    else:
        flagged = res["flagged"]
        status = "⚠️ Potentially Unreliable" if flagged else "🟢 Likely Reliable"
        c = st.columns(4)
        with c[0]: big("Status", status, f"warning at ≥ {res['threshold']:.0%} risk (threshold chosen on validation data)",
                       "status-warn" if flagged else "status-ok")
        with c[1]: big("Estimated error risk", pct(res["risk"], 0), "estimated chance the prediction is incorrect")
        with c[2]: big("Prediction stability", pct(res["stability"], 0), "of 6 slightly changed copies keep the same prediction")
        with c[3]: big("Model", "Random Forest", "reliability model (confidence, image quality, stability)")
        st.caption("The reliability model estimates the risk that the prediction may be incorrect. It is an estimate, not a guarantee.")

    step(4, "🔬 Automatic robustness check")
    st.markdown("The system automatically tests the prediction under different image conditions "
                "(moderate strength from the project's evaluation settings).")
    rows = "".join(
        f"<tr><td>{html.escape(r['condition'])}</td><td>{html.escape(pretty(r['pred_class']))}</td><td>{r['confidence']:.1%}</td>"
        f"<td>{'✅ same' if r['same_as_original'] else '⚠️ changed'}</td></tr>" for r in rob)
    st.markdown(f'<table class="robust"><thead><tr><th>Condition</th><th>Prediction</th><th>Confidence</th><th>vs original</th></tr>'
                f'</thead><tbody>{rows}</tbody></table>', unsafe_allow_html=True)
    agree = sum(r["same_as_original"] for r in rob)
    st.markdown(f"**Prediction agreement: {agree} / {len(rob)}** conditions give the same disease as the original image.")
    if agree == len(rob):
        banner("ok", "🟢 <b>Prediction remained stable under the tested conditions.</b>")
    else:
        banner("warn", "⚠️ <b>Prediction became unstable under the tested conditions.</b> The prediction changed under image "
                       "degradation, indicating reduced prediction stability. This alone does not show which prediction is correct.")

    step(5, "📊 Final result")
    rel_status = ("Potentially Unreliable" if res["flagged"] else "Likely Reliable") if "risk" in res else "not available"
    final_rows = [("Disease", pretty(res["pred_class"])), ("Confidence", pct(res["confidence"])), ("Reliability", rel_status)]
    if "risk" in res:
        final_rows.append(("Estimated error risk", pct(res["risk"], 0)))
    final_rows.append(("Robustness stability", f"{agree} / {len(rob)} conditions agree"))
    st.markdown('<div class="final">' + "".join(f'<div class="row"><span class="k">{html.escape(k)}</span><span class="v">{html.escape(v)}</span></div>'
                                              for k, v in final_rows) + "</div>", unsafe_allow_html=True)

    st.markdown("### How it works")
    st.markdown("- **EfficientNet-B0** identifies the plant disease.\n"
                "- The system automatically checks how **stable** the prediction remains under changed image conditions.\n"
                "- A **Random Forest reliability model** estimates the risk that the prediction may be incorrect.")

# ---------------------------------------------------------------- Model Performance (technical evidence)
elif page == "Model Performance":
    hero("Model performance", "Measured results on the locked PlantVillage test set",
         "Every number below is read from reports/metrics. The test images were never used for training or tuning.")
    clf, rel = read_json("classifier_test_metrics.json"), read_json("reliability_test_metrics.json")
    if not clf:
        need_setup("Evaluation results")
        st.stop()
    t1, t2, t3 = st.tabs(["Classifier", "Reliability model", "Robustness"])

    with t1:
        c = st.columns(3)
        with c[0]: card("Test accuracy", pct(clf["accuracy"], 2), "clean held-out images")
        with c[1]: card("Macro F1-score", f"{clf['macro_f1']:.4f}", "average over classes")
        with c[2]: card("Test images", f"{clf['test_images']:,}", f"{clf['num_classes']} classes")
        c = st.columns(3)
        with c[0]: card("Macro precision", f"{clf['macro_precision']:.4f}")
        with c[1]: card("Macro recall", f"{clf['macro_recall']:.4f}")
        with c[2]: card("Model size", f"{clf['model_file_mb']:.1f} MB", f"{clf['parameters']:,} parameters")
        cm = pd.read_csv(METRICS / "test_confusion_matrix.csv", index_col=0)
        norm = cm.div(cm.sum(axis=1).clip(lower=1), axis=0)
        labels = [pretty(x) for x in cm.index]
        fig = go.Figure(go.Heatmap(z=norm.values, x=labels, y=labels, colorscale="Greens", zmin=0, zmax=1,
                                   hovertemplate="true %{y}<br>pred %{x}<br>%{z:.1%}<extra></extra>"))
        fig.update_layout(height=760, title=f"Row-normalised confusion matrix (clean test set, n={clf['test_images']:,})",
                          xaxis=dict(tickfont=dict(size=8)), yaxis=dict(tickfont=dict(size=8), autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)
        per_class = read_csv("test_per_class_metrics.csv")
        if per_class is not None:
            st.markdown("**Per-class metrics**")
            st.dataframe(per_class.round(4), use_container_width=True)

    with t2:
        if not rel:
            need_setup("Reliability results")
        else:
            st.markdown("Positive class = **incorrect** disease prediction. Test rows: each test image clean plus 3 sampled degradations "
                        f"({rel['test_rows']:,} rows). Thresholds were chosen on validation data.")
            m = rel["methods"]
            rb, cb = m["Random Forest (B)"]["all conditions"], m["Confidence threshold"]["all conditions"]
            c = st.columns(4)
            with c[0]: card("PR-AUC (Random Forest B)", f"{rb['pr_auc']:.3f}", f"confidence baseline {cb['pr_auc']:.3f}")
            with c[1]: card("ROC-AUC (Random Forest B)", f"{rb['roc_auc']:.3f}", f"confidence baseline {cb['roc_auc']:.3f}")
            with c[2]: card("F1 (Random Forest B)", f"{rb['f1']:.3f}", f"confidence baseline {cb['f1']:.3f}")
            with c[3]: card("Recall of errors (RF B)", pct(rb["recall"]), f"confidence baseline {pct(cb['recall'])}")
            cmp_ = read_csv("reliability_baseline_comparison.csv")
            st.dataframe(cmp_.round(4), hide_index=True, use_container_width=True)
            fig = go.Figure()
            for name, r in m.items():
                fig.add_trace(go.Scatter(x=r["risk_coverage"]["coverage"], y=r["risk_coverage"]["error_rate"], name=name, mode="lines"))
            fig.update_layout(height=340, title="Risk vs coverage (lower is better)", xaxis_title="Share of predictions accepted",
                              yaxis_title="Error rate among accepted", yaxis_tickformat=".1%")
            st.plotly_chart(fig, use_container_width=True)
            cal = pd.DataFrame(rel["calibration_B"]["bins"])
            fig = go.Figure([go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dot", color="#999"), name="perfect"),
                             go.Scatter(x=cal["mean_predicted"], y=cal["observed_incorrect"], mode="lines+markers", name="Random Forest B")])
            fig.update_layout(height=320, title=f"Calibration of the estimated error risk (Brier {rel['calibration_B']['brier']:.4f})",
                              xaxis_title="Estimated error risk", yaxis_title="Observed share incorrect")
            st.plotly_chart(fig, use_container_width=True)
            imp = read_csv("reliability_feature_importance_B.csv")
            if imp is not None:
                st.plotly_chart(px.bar(imp.sort_values("permutation_importance_pr_auc"), x="permutation_importance_pr_auc", y="feature",
                                       orientation="h", height=460, title="Feature importance (Random Forest B, permutation, validation data)"),
                                use_container_width=True)

    with t3:
        rob = read_csv("robustness_by_corruption.csv")
        d = rob[rob["family"] != "clean"]
        fig = px.line(d, x="severity", y="accuracy", color="family", markers=True)
        fig.add_hline(y=clf["accuracy"], line_dash="dot", line_color=FOREST, annotation_text=f"clean {clf['accuracy']:.1%}")
        fig.update_layout(height=360, yaxis_tickformat=".0%", title="Test accuracy vs degradation strength",
                          xaxis=dict(tickvals=[1, 2, 3], title="Severity (1 = mild, 3 = strong)"))
        st.plotly_chart(fig, use_container_width=True)
        fig = px.bar(d, x="family", y="accuracy", color=d["severity"].astype(str), barmode="group",
                     labels={"color": "severity"}, color_discrete_sequence=["#9cc58a", LEAF, FOREST])
        fig.update_layout(height=360, yaxis_tickformat=".0%", title="Test accuracy by degradation type and severity")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(rob.drop(columns=["inference_ms_per_pass"]).round(4), hide_index=True, use_container_width=True)

# ---------------------------------------------------------------- About
else:
    hero("About", "AgriShield", "Plant disease detection that also estimates when its own prediction may be wrong.")
    st.markdown("""
**Objective.** Classify plant leaf diseases from a photo and, alongside each prediction, estimate the risk that it is incorrect,
especially when the photo is dark, blurred, noisy, compressed or low-resolution.

**Data.** PlantVillage (Mohanty, Hughes & Salathé, 2016; CC BY-SA 3.0): 54,305 leaf images in 38 classes, split into training,
validation, calibration and a locked test set. Duplicate and same-leaf photos always stay in the same split.

**EfficientNet-B0.** A compact convolutional network (about 4 million parameters), pre-trained on ImageNet and fine-tuned on PlantVillage.
It outputs a probability for each of the 38 classes. The highest is the prediction, and its probability is the confidence.

**Random Forest reliability model.** Trained on a separate calibration split to predict whether the classifier's prediction is
incorrect. Its inputs are the classifier's confidence values, simple image-quality measures (brightness, contrast, sharpness, noise)
and the prediction's stability.

**Automatic reliability checking.** For every uploaded image the app makes 6 slightly changed copies (a little darker or brighter,
slight blur, light noise, JPEG compression, lower resolution). It checks whether the prediction stays the same, then feeds everything
to the Random Forest. The robustness check on the main page separately shows how the prediction behaves under stronger degradations.

**Contribution.** A complete, leakage-aware pipeline that combines a standard classifier with a learned error-risk estimate, evaluated
under controlled image degradations on a locked test set. No new algorithm is claimed.

**Limitations.**
- PlantVillage photos show a single leaf on a plain background.
- Performance drops substantially on real field photos (different backgrounds, several leaves, lighting). This domain shift was
  measured in an external evaluation, documented in the project's technical report.
- The estimated error risk is learned from PlantVillage-style errors and is not a guarantee.

**Disclaimer.** This is a research demonstration, not a replacement for diagnosis by a qualified agricultural expert.
See LICENSES_AND_ATTRIBUTION.md for dataset and software licences.
""")
