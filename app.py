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

    with st.spinner("Analysing the image: disease prediction, reliability assessment and robustness test ..."):
        res = predictor.analyse(img, with_stability=True)        # EfficientNet + 6 stability copies + Random Forest B
        rob = robustness_demo.run(predictor, img)                 # original + 5 representative degradations

    step(1, "🖼️ Uploaded image")
    c_img, _ = st.columns([2, 3])
    c_img.image(img, caption=name, width="stretch")

    step(2, "🌿 Disease prediction")
    c = st.columns(2)
    with c[0]: big("Predicted disease", pretty(res["pred_class"]), "identified by EfficientNet-B0")
    with c[1]: big("Confidence", pct(res["confidence"]), "the model's probability for this class (not accuracy)")

    step(3, "🛡️ Reliability assessment")
    if "risk" not in res:
        need_setup("The reliability model")
        flagged = None
    else:
        flagged = res["flagged"]
        c = st.columns(2)
        with c[0]: big("Status", "⚠️ Potentially Unreliable" if flagged else "🟢 Reliable",
                       "reliability assessment by a Random Forest model", "status-warn" if flagged else "status-ok")
        with c[1]: big("Estimated error risk", pct(res["risk"], 0), "estimated chance that the prediction is incorrect")
        st.caption("The reliability model estimates the risk that the prediction may be incorrect. It is an estimate, not a guarantee.")

    step(4, "🔬 Automatic robustness test")
    st.markdown("The image is automatically changed in 5 ways and classified again. The original prediction is the reference.")
    degraded = rob[1:]
    agree = sum(r["same_as_original"] for r in degraded)
    rows = f"<tr><td>{html.escape(rob[0]['condition'])}</td><td>{html.escape(pretty(rob[0]['pred_class']))}</td><td>Reference</td></tr>"
    rows += "".join(f"<tr><td>{html.escape(r['condition'])}</td><td>{html.escape(pretty(r['pred_class']))}</td>"
                    f"<td>{'✓' if r['same_as_original'] else '✗'}</td></tr>" for r in degraded)
    st.markdown(f'<table class="robust"><thead><tr><th>Condition</th><th>Prediction</th><th>Agreement</th></tr></thead>'
                f'<tbody>{rows}</tbody></table>', unsafe_allow_html=True)
    st.markdown(f"**Prediction agreement: {agree}/{len(degraded)} degraded conditions**")
    if agree < len(degraded):
        st.caption("The prediction changed under image degradation, indicating reduced prediction stability. "
                   "This alone does not show which prediction is correct.")

    step(5, "📊 Final assessment")
    reliable = flagged is False and agree == len(degraded)
    if reliable:
        banner("ok", "🟢 <b>Prediction appears reliable.</b>")
    else:
        banner("warn", "⚠️ <b>Prediction may be unreliable under changed image conditions.</b>")

    st.markdown("### How it works")
    st.markdown("- **EfficientNet-B0** identifies the plant disease.\n"
                "- A **Random Forest reliability model** estimates the risk that the prediction may be incorrect.\n"
                "- The **automatic robustness test** checks whether the prediction stays the same when the image is darker, "
                "blurred, noisy, compressed or low-resolution.")

# ---------------------------------------------------------------- Model Performance (technical evidence)
elif page == "Model Performance":
    hero("Model performance", "Measured results on the locked PlantVillage test set",
         "Every number below is read from reports/metrics. The test images were never used for training or tuning.")
    clf, rel = read_json("classifier_test_metrics.json"), read_json("reliability_test_metrics.json")
    if not clf:
        need_setup("Evaluation results")
        st.stop()
    ext = read_json("external_plantdoc_metrics.json")
    t1, t2, t3, t4 = st.tabs(["Classifier", "Robustness", "Reliability model", "External PlantDoc evaluation"])

    with t1:
        st.markdown(f"**Model:** EfficientNet-B0 · **Dataset:** PlantVillage · **Evaluation:** locked held-out test set · "
                    f"**Test images:** {clf['test_images']:,} · **Classes:** {clf['num_classes']}")
        c = st.columns(4)
        with c[0]: card("Accuracy", pct(clf["accuracy"], 2), "clean held-out images")
        with c[1]: card("Precision", f"{clf['macro_precision']:.4f}", "macro average")
        with c[2]: card("Recall", f"{clf['macro_recall']:.4f}", "macro average")
        with c[3]: card("F1-score", f"{clf['macro_f1']:.4f}", "macro average")
        cm = pd.read_csv(METRICS / "test_confusion_matrix.csv", index_col=0)
        norm = cm.div(cm.sum(axis=1).clip(lower=1), axis=0)
        labels = [pretty(x) for x in cm.index]
        fig = go.Figure(go.Heatmap(z=norm.values, x=labels, y=labels, colorscale="Greens", zmin=0, zmax=1,
                                   hovertemplate="true %{y}<br>pred %{x}<br>%{z:.1%}<extra></extra>"))
        fig.update_layout(height=760, title=f"Row-normalised confusion matrix (clean test set, n={clf['test_images']:,})",
                          xaxis=dict(tickfont=dict(size=8)), yaxis=dict(tickfont=dict(size=8), autorange="reversed"))
        st.plotly_chart(fig, width="stretch")
        per_class = read_csv("test_per_class_metrics.csv")
        if per_class is not None:
            st.markdown("**Per-class metrics**")
            per_class = per_class.rename(columns={per_class.columns[0]: "Class"})
            per_class["Class"] = per_class["Class"].map(pretty)
            st.dataframe(per_class.round(4), hide_index=True, width="stretch")

    with t2:
        rob = read_csv("robustness_by_corruption.csv")
        d = rob[rob["family"] != "clean"]
        worst = d.sort_values("accuracy").iloc[0]
        banner("warn", f"<b>Clean accuracy is high, but it drops as image quality drops.</b> On clean test images the classifier "
                       f"reached {clf['accuracy']:.1%}. On the same images with {worst['family'].replace('_', ' ')} at the strongest "
                       f"level, accuracy was {worst['accuracy']:.1%}. This is why the prediction's reliability needs checking.")
        fig = px.line(d, x="severity", y="accuracy", color="family", markers=True)
        fig.add_hline(y=clf["accuracy"], line_dash="dot", line_color=FOREST, annotation_text=f"clean {clf['accuracy']:.1%}")
        fig.update_layout(height=360, yaxis_tickformat=".0%", title="Test accuracy vs degradation strength",
                          xaxis=dict(tickvals=[1, 2, 3], title="Severity (1 = mild, 3 = strong)"))
        st.plotly_chart(fig, width="stretch")
        fig = px.bar(d, x="family", y="accuracy", color=d["severity"].astype(str), barmode="group",
                     labels={"color": "severity"}, color_discrete_sequence=["#9cc58a", LEAF, FOREST])
        fig.update_layout(height=360, yaxis_tickformat=".0%", title="Test accuracy by degradation type and severity")
        st.plotly_chart(fig, width="stretch")
        st.dataframe(rob.drop(columns=["inference_ms_per_pass"]).round(4), hide_index=True, width="stretch")

    with t3:
        if not rel:
            need_setup("Reliability results")
        else:
            st.markdown("The **Random Forest** is a secondary ML model. It estimates the risk that EfficientNet's prediction is incorrect, "
                        "using the classifier's confidence, simple image-quality measures and how stable the prediction is under small "
                        "changes. Positive class = **incorrect** prediction. Test rows are each test image clean plus 3 sampled "
                        f"degradations ({rel['test_rows']:,} rows). Thresholds were chosen on validation data.")
            m = rel["methods"]
            rb, cb = m["Random Forest (B)"]["all conditions"], m["Confidence threshold"]["all conditions"]
            c = st.columns(5)
            with c[0]: card("Precision", f"{rb['precision']:.3f}", f"confidence baseline {cb['precision']:.3f}")
            with c[1]: card("Recall", f"{rb['recall']:.3f}", f"confidence baseline {cb['recall']:.3f}")
            with c[2]: card("F1", f"{rb['f1']:.3f}", f"confidence baseline {cb['f1']:.3f}")
            with c[3]: card("ROC-AUC", f"{rb['roc_auc']:.3f}", f"confidence baseline {cb['roc_auc']:.3f}")
            with c[4]: card("PR-AUC", f"{rb['pr_auc']:.3f}", f"confidence baseline {cb['pr_auc']:.3f}")
            st.markdown("**Comparison with simple baselines** (flag a prediction when confidence, or the gap between the top two classes, is low)")
            cmp_ = read_csv("reliability_baseline_comparison.csv")
            st.table(cmp_.set_index("method")[["precision", "recall", "f1", "roc_auc", "pr_auc", "false_alarm_rate"]].rename(
                columns={"f1": "F1", "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC", "false_alarm_rate": "false alarms"}).round(3))
            fig = go.Figure()
            for name, r in m.items():
                fig.add_trace(go.Scatter(x=r["risk_coverage"]["coverage"], y=r["risk_coverage"]["error_rate"], name=name, mode="lines"))
            fig.update_layout(height=340, title="Risk vs coverage (lower is better)", xaxis_title="Share of predictions accepted",
                              yaxis_title="Error rate among accepted", yaxis_tickformat=".1%")
            st.plotly_chart(fig, width="stretch")
            cal = pd.DataFrame(rel["calibration_B"]["bins"])
            fig = go.Figure([go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dot", color="#999"), name="perfect"),
                             go.Scatter(x=cal["mean_predicted"], y=cal["observed_incorrect"], mode="lines+markers", name="Random Forest B")])
            fig.update_layout(height=320, title=f"Calibration of the estimated error risk (Brier {rel['calibration_B']['brier']:.4f})",
                              xaxis_title="Estimated error risk", yaxis_title="Observed share incorrect")
            st.plotly_chart(fig, width="stretch")
            imp = read_csv("reliability_feature_importance_B.csv")
            if imp is not None:
                st.plotly_chart(px.bar(imp.sort_values("permutation_importance_pr_auc"), x="permutation_importance_pr_auc", y="feature",
                                       orientation="h", height=460, title="Feature importance (Random Forest B, permutation, validation data)"),
                                width="stretch")

    with t4:
        if not ext:
            need_setup("External PlantDoc evaluation")
        else:
            st.markdown("**External PlantDoc Evaluation.** PlantDoc contains field-style photos (cluttered backgrounds, several leaves, "
                        "varied lighting), while PlantVillage contains single leaves on plain backgrounds. The model was **never trained** on "
                        "PlantDoc, and only classes that match PlantVillage labels were used.")
            s_, x_ = ext["strict_exact_mappings"], ext["extended_with_ambiguous"]
            c = st.columns(4)
            with c[0]: card("Accuracy", pct(s_["accuracy"]), f"{s_['images']:,} images, exact label matches")
            with c[1]: card("Top-3 accuracy", pct(s_["top3_accuracy"]), "correct class among the top 3")
            with c[2]: card("Macro F1-score", f"{s_['macro_f1_over_present_classes']:.3f}", f"{s_['classes']} classes")
            with c[3]: card("With ambiguous matches", pct(x_["accuracy"]), f"{x_['images']:,} images")
            banner("warn", f"<b>Domain shift.</b> The same classifier reached {clf['accuracy']:.1%} on the PlantVillage test set but "
                           f"{s_['accuracy']:.1%} on PlantDoc. Accuracy on controlled images does not transfer to field-style images.")
            pc_ = pd.DataFrame(ext["per_class_accuracy_strict"]).sort_values("accuracy")
            st.plotly_chart(px.bar(pc_, x="accuracy", y="plantdoc_class", orientation="h", height=620, hover_data=["images"],
                                   title="PlantDoc accuracy per class (exact label matches)").update_layout(xaxis_tickformat=".0%"),
                            width="stretch")

# ---------------------------------------------------------------- About
else:
    hero("About", "AgriShield", "Plant disease detection that also checks how reliable each prediction looks when image conditions change.")
    st.markdown("""
AgriShield is a plant disease detection system. It does not only predict the disease; it also evaluates how reliable that prediction
appears when the image conditions change.

**Pipeline**

Image → **EfficientNet-B0** → Disease prediction → **Reliability assessment** → **Automatic robustness testing** → **Final assessment**

**Models**
- **EfficientNet-B0:** disease classification. A compact convolutional network pre-trained on ImageNet and fine-tuned on
  PlantVillage leaf images (38 classes).
- **Random Forest:** prediction error-risk estimation. A secondary model, trained on a separate calibration split. It estimates the
  risk that EfficientNet's prediction is incorrect, using the classifier's confidence, simple image-quality measures and how stable
  the prediction stays under small image changes.

**Data and evaluation**
- **PlantVillage:** 54,305 images, 38 classes; split into training, validation, calibration and a locked test set.
  Duplicate and same-leaf photos always stay in the same split.
- **PlantDoc:** field-style photos used only for external evaluation, never for training.

**Contribution.** This project presents a reliability-aware evaluation framework that combines plant disease classification,
automatic image-degradation testing, and a learned error-risk detector. It uses established models (EfficientNet-B0 and a Random Forest).
It does not introduce a new algorithm.

**Limitations**
- PlantVillage photos show a single leaf on a plain background. Accuracy is much lower on field-style photos (see Model Performance →
  External PlantDoc evaluation).
- The estimated error risk is learned from PlantVillage-style errors. It is an estimate, not a guarantee.

**Disclaimer.** This is a research demonstration, not a replacement for diagnosis by a qualified agricultural expert.
Data: PlantVillage (Mohanty, Hughes & Salathé, 2016; CC BY-SA 3.0) and PlantDoc (Singh et al., 2020; CC BY 4.0).
""")
