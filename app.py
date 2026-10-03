"""AgriShield - reliability-aware plant disease detection (research demonstration).

Run:  streamlit run app.py      (or double-click run_app.bat on Windows)
Every metric shown is read from reports/metrics/*. Nothing is retrained when the app starts.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from src import perturbations  # noqa: E402
from src.imaging import ImageError, load_rgb, standardise, validate_upload  # noqa: E402
from src.model import ModelFileError  # noqa: E402
from src.predictor import Predictor, pretty  # noqa: E402
from src.ui import AMBER, FOREST, LEAF, RED, banner, card, hero, inject_css, pct  # noqa: E402

st.set_page_config(page_title="AgriShield", page_icon="🌿", layout="wide")
inject_css()
METRICS = ROOT / "reports" / "metrics"
SAMPLES = ROOT / "assets" / "samples"
QUALITY_LABELS = {"brightness": "Brightness (0-1)", "contrast": "Contrast (luma std)", "sharpness_log_lapvar": "Sharpness (log Laplacian variance)",
                  "noise_sigma": "Estimated noise", "saturation": "Colour saturation", "original_min_side": "Shorter side (px)"}


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


def quality_reference():
    p = ROOT / "models" / "quality_reference.json"
    return json.loads(p.read_text()) if p.exists() else None


def need_setup(what: str) -> None:
    banner("warn", f"<b>{what} not available yet.</b> Run the training and evaluation steps (see README: "
                   f"<code>train_windows.bat</code> then <code>evaluate_windows.bat</code>). No numbers are shown until they are measured.")


def image_input(key: str):
    """Upload or pick a bundled sample. Returns (PIL image, name) or (None, None)."""
    samples = sorted(SAMPLES.glob("*.jpg")) + sorted(SAMPLES.glob("*.png"))
    c1, c2 = st.columns([3, 2])
    up = c1.file_uploader("Upload a leaf photo (JPG, JPEG or PNG, up to 15 MB)", type=["jpg", "jpeg", "png"], key=f"up_{key}")
    choice = c2.selectbox("…or try a sample image", ["—"] + [s.name for s in samples], key=f"sample_{key}")
    try:
        if up is not None:
            validate_upload(up.name, up.size)
            return load_rgb(up), up.name
        if choice != "—":
            return load_rgb(SAMPLES / choice), choice
    except ImageError as e:
        banner("bad", f"<b>Could not use this file.</b> {e}")
    return None, None


def analyse(pred: Predictor, img, stability: bool):
    with st.spinner("Running EfficientNet-B0" + (" and the 6-pass stability check" if stability else "") + " ..."):
        return pred.analyse(img, with_stability=stability)


def risk_gauge(risk: float, threshold: float, title: str = "Estimated reliability risk"):
    fig = go.Figure(go.Indicator(mode="gauge+number", value=risk * 100, number={"suffix": "%", "font": {"size": 34, "color": FOREST}},
                                 title={"text": title, "font": {"size": 14}},
                                 gauge={"axis": {"range": [0, 100]}, "bar": {"color": RED if risk >= threshold else LEAF},
                                        "threshold": {"line": {"color": AMBER, "width": 4}, "value": threshold * 100},
                                        "bgcolor": "#ffffff"}))
    fig.update_layout(height=230, margin=dict(l=20, r=20, t=50, b=10))
    return fig


def top3_chart(top3):
    df = pd.DataFrame({"class": [pretty(c) for c, _ in top3][::-1], "p": [p for _, p in top3][::-1]})
    fig = px.bar(df, x="p", y="class", orientation="h", text=df["p"].map(lambda v: f"{v:.1%}"))
    fig.update_traces(marker_color=[LEAF, LEAF, FOREST], textposition="outside", cliponaxis=False)
    fig.update_layout(height=200, xaxis=dict(range=[0, 1.1], tickformat=".0%", title="Model probability"), yaxis_title=None)
    return fig


def quality_flags(feats: dict) -> list[str]:
    """Observations only: which indicators fall outside the 5th-95th percentile of clean calibration images."""
    ref = quality_reference()
    if not ref:
        return []
    notes = []
    for k, label in QUALITY_LABELS.items():
        if k in ref and k in feats:
            lo, hi = ref[k]["p05"], ref[k]["p95"]
            if feats[k] < lo:
                notes.append(f"{label} is <b>lower</b> than in 95% of clean training-style images ({feats[k]:.3g} vs ≥ {lo:.3g}).")
            elif feats[k] > hi:
                notes.append(f"{label} is <b>higher</b> than in 95% of clean training-style images ({feats[k]:.3g} vs ≤ {hi:.3g}).")
    return notes


def reliability_banner(res: dict) -> None:
    if "risk" not in res:
        banner("warn", "Reliability model not trained yet - only the disease prediction is shown.")
        return
    if res["flagged"]:
        banner("bad", f"<b>Potentially unreliable.</b> Estimated risk {res['risk']:.0%} is at or above the validation-selected "
                      f"threshold of {res['threshold']:.0%}. Treat this prediction with caution and consult an expert.")
    else:
        banner("ok", f"<b>No reliability warning.</b> Estimated risk {res['risk']:.0%} is below the validation-selected threshold "
                     f"of {res['threshold']:.0%}. This is not a guarantee that the prediction is correct.")


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("## 🌿 AgriShield")
    st.markdown("Reliability-aware plant disease detection")
    page = st.radio("Navigate", ["Dashboard", "Disease Detection", "Robustness Lab", "Reliability Report", "Model Evaluation", "About"],
                    label_visibility="collapsed")
    st.markdown('<div class="disclaimer">Research demonstration only - not a substitute for diagnosis by an agricultural expert.</div>',
                unsafe_allow_html=True)

clf = read_json("classifier_test_metrics.json")
rel = read_json("reliability_test_metrics.json")
ext = read_json("external_plantdoc_metrics.json")
data_rep = read_json("dataset_report.json")

predictor, load_error = None, None
try:
    predictor = get_predictor()
except (ModelFileError, Exception) as e:   # shown to the user; the app still renders the static pages
    load_error = str(e) if isinstance(e, ModelFileError) else f"The model could not be loaded ({e.__class__.__name__}: {e})."

# ---------------------------------------------------------------- pages
if page == "Dashboard":
    hero("Research dashboard", "Can we tell when the disease model is wrong?",
         "EfficientNet-B0 classifies 38 PlantVillage leaf conditions. A Random Forest then estimates the risk that each prediction is "
         "incorrect, using confidence, image quality and stability under small perturbations.")
    if not clf:
        need_setup("Evaluation results")
    else:
        st.markdown('<span class="section-tag">INTERNAL TEST · PlantVillage locked test set</span>', unsafe_allow_html=True)
        c = st.columns(4)
        with c[0]: card("Test accuracy", pct(clf["accuracy"]), "clean held-out images")
        with c[1]: card("Macro F1-score", f"{clf['macro_f1']:.3f}", "average over classes")
        with c[2]: card("Disease classes", str(clf["num_classes"]), "PlantVillage labels")
        with c[3]: card("Test images", f"{clf['test_images']:,}", "never used for training or tuning")
        if rel:
            st.markdown('<span class="section-tag">RELIABILITY · positive = incorrect prediction · test set: clean + 3 sampled degradations per image</span>',
                        unsafe_allow_html=True)
            m = rel["methods"]
            rb, cb = m["Random Forest (B)"]["all conditions"], m["Confidence threshold"]["all conditions"]
            c = st.columns(4)
            with c[0]: card("Recall of errors (RF-B)", pct(rb["recall"]), f"confidence baseline {pct(cb['recall'])}")
            with c[1]: card("Precision of warnings (RF-B)", pct(rb["precision"]), f"confidence baseline {pct(cb['precision'])}")
            with c[2]: card("PR-AUC (RF-B)", f"{rb['pr_auc']:.3f}", f"confidence baseline {cb['pr_auc']:.3f}")
            with c[3]: card("Test rows", f"{rel['test_rows']:,}", f"{rel['test_images']:,} images × {rel['conditions_per_image']} conditions")
        if ext:
            s = ext["strict_exact_mappings"]
            st.markdown('<span class="section-tag">EXTERNAL · PlantDoc field photos (different dataset)</span>', unsafe_allow_html=True)
            c = st.columns(4)
            with c[0]: card("External accuracy", pct(s["accuracy"]), "exact label mappings only")
            with c[1]: card("External top-3 accuracy", pct(s["top3_accuracy"]), "")
            with c[2]: card("External images", f"{s['images']:,}", f"{s['classes']} mapped classes")
            with c[3]: card("Macro F1 (present classes)", f"{s['macro_f1_over_present_classes']:.3f}", "")
        rob = read_csv("robustness_by_corruption.csv")
        if rob is not None:
            st.markdown("### Accuracy under image degradation")
            d = rob[rob["family"] != "clean"]
            fig = px.line(d, x="severity", y="accuracy", color="family", markers=True)
            fig.add_hline(y=clf["accuracy"], line_dash="dot", line_color=FOREST, annotation_text="clean accuracy")
            fig.update_layout(height=340, yaxis_tickformat=".0%", xaxis=dict(tickvals=[1, 2, 3], title="Severity (1 = mild, 3 = strong)"))
            st.plotly_chart(fig, use_container_width=True)

elif page == "Disease Detection":
    hero("Disease detection", "Upload a leaf photo", "Get the predicted disease, the top-3 alternatives, the model's confidence and an "
         "estimated reliability risk.")
    if load_error:
        banner("bad", f"<b>Model not loaded.</b> {load_error}")
        st.stop()
    img, name = image_input("detect")
    stability = st.toggle("Run the stability check (6 extra forward passes, used by reliability model B)", value=True)
    if img is not None:
        res = analyse(predictor, img, stability)
        st.session_state["last_result"], st.session_state["last_name"] = res, name
        c1, c2 = st.columns([2, 3])
        c1.image(img, caption=f"{name} · {img.width}×{img.height}px", use_container_width=True)
        with c2:
            st.markdown(f"### {pretty(res['pred_class'])}")
            st.markdown(f'<span class="pill ok">Prediction confidence {res["confidence"]:.1%}</span> '
                        f'<span class="small">model probability for this class on this image - not accuracy</span>', unsafe_allow_html=True)
            st.plotly_chart(top3_chart(res["top3"]), use_container_width=True)
            reliability_banner(res)
        if "risk" in res:
            g1, g2 = st.columns(2)
            g1.plotly_chart(risk_gauge(res["risk"], res["threshold"], f"Reliability risk (model {res['reliability_config']})"), use_container_width=True)
            with g2:
                if "stability" in res:
                    card("Prediction stability", pct(res["stability"], 0), "share of 6 perturbed copies with the same prediction")
                st.markdown("")
                card("Validation-selected threshold", pct(res["threshold"], 0), "chosen on reliability-validation data (F2)")
        with st.expander("Image-quality indicators"):
            f = res["features"]
            st.dataframe(pd.DataFrame({"indicator": [QUALITY_LABELS[k] for k in QUALITY_LABELS], "value": [round(f[k], 4) for k in QUALITY_LABELS]}),
                         hide_index=True, use_container_width=True)

elif page == "Robustness Lab":
    hero("Robustness lab", "Degrade an image and watch the prediction",
         "Change brightness, blur, noise, compression and resolution. The original and the altered image go through the same model.")
    if load_error:
        banner("bad", f"<b>Model not loaded.</b> {load_error}")
        st.stop()
    img, name = image_input("lab")
    if img is not None:
        s = st.columns(5)
        bright = s[0].slider("Brightness", 0.2, 2.0, 1.0, 0.05)
        blur = s[1].slider("Blur (sigma px)", 0.0, 5.0, 0.0, 0.25)
        noise = s[2].slider("Noise (std)", 0.0, 0.3, 0.0, 0.01)
        jpeg = s[3].slider("JPEG quality", 5, 100, 100, 5)
        res_scale = s[4].slider("Resolution scale", 0.1, 1.0, 1.0, 0.05)
        std_img = standardise(img)
        altered = std_img
        if bright != 1.0:
            altered = perturbations.apply(altered, "brightness_low" if bright < 1 else "brightness_high", bright)
        if blur > 0:
            altered = perturbations.apply(altered, "gaussian_blur", blur)
        if noise > 0:
            altered = perturbations.apply(altered, "gaussian_noise", noise, seed=0)
        if jpeg < 100:
            altered = perturbations.apply(altered, "jpeg_compression", jpeg)
        if res_scale < 1.0:
            altered = perturbations.apply(altered, "low_resolution", res_scale)
        r0, r1 = analyse(predictor, std_img, True), analyse(predictor, altered, True)
        c1, c2 = st.columns(2)
        for col, title, im, r in ((c1, "Original", std_img, r0), (c2, "Altered", altered, r1)):
            with col:
                st.markdown(f"#### {title}")
                st.image(im, use_container_width=True)
                st.markdown(f"**{pretty(r['pred_class'])}**  \nConfidence {r['confidence']:.1%}"
                            + (f" · reliability risk {r['risk']:.0%}" if "risk" in r else "")
                            + (f" · stability {r['stability']:.0%}" if "stability" in r else ""))
        if r0["pred_index"] != r1["pred_index"]:
            banner("bad", "<b>The prediction changed.</b> The degradation alone was enough to change the disease label, so at least one of "
                          "the two predictions is wrong. Use the altered result with caution.")
        else:
            banner("ok", "<b>Prediction unchanged</b> by these settings.")
        if r1.get("flagged"):
            banner("warn", "<b>The altered image is potentially unreliable</b> according to the reliability model.")
        st.session_state["last_result"], st.session_state["last_name"] = r1, f"{name} (altered)"

elif page == "Reliability Report":
    hero("Reliability report", "Why did (or didn't) a warning appear?",
         "Details for the most recent image analysed on the Detection or Robustness page.")
    res = st.session_state.get("last_result")
    if res is None:
        banner("warn", "Analyse an image on the <b>Disease Detection</b> or <b>Robustness Lab</b> page first.")
    elif "risk" not in res:
        need_setup("The reliability model")
    else:
        st.markdown(f"**Image:** {st.session_state.get('last_name')} · **Prediction:** {pretty(res['pred_class'])}")
        thr = st.slider("Warning threshold (default = value selected on validation data)", 0.0, 1.0, float(res["threshold"]), 0.01)
        flagged = res["risk"] >= thr
        c = st.columns(4)
        with c[0]: card("Estimated risk of error", pct(res["risk"], 0), f"Random Forest model {res['reliability_config']}")
        with c[1]: card("Status", "Potentially unreliable" if flagged else "Reliable", f"threshold {thr:.0%}")
        with c[2]: card("Prediction confidence", pct(res["confidence"]), "classifier probability")
        with c[3]: card("Stability", pct(res["stability"], 0) if "stability" in res else "n/a", "of 6 perturbed copies agree")
        st.markdown("#### What we observed")
        notes = []
        f = res["features"]
        if f["max_prob"] < 0.6:
            notes.append(f"The classifier is not confident: its top probability is {f['max_prob']:.0%}.")
        if f["margin_top1_top2"] < 0.2:
            notes.append(f"The top two classes are close (gap {f['margin_top1_top2']:.0%}).")
        if "stability" in res and res["stability"] < 1:
            notes.append(f"The prediction changed for {round((1 - res['stability']) * 6)} of the 6 small perturbations.")
        notes += quality_flags(f)
        if notes:
            st.markdown("\n".join(f"- {n}" for n in notes), unsafe_allow_html=True)
        else:
            st.markdown("- Confidence, stability and image-quality indicators all look typical of clean test images.")
        st.caption("These are observations of the inputs to the Random Forest, not a proven cause of an error. The risk is a score from a "
                   "model trained on PlantVillage; check the calibration chart under Model Evaluation before treating it as a probability.")

elif page == "Model Evaluation":
    hero("Model evaluation", "Measured results", "Everything below is read from reports/metrics. Internal (PlantVillage) and external "
         "(PlantDoc) results are kept separate.")
    if not clf:
        need_setup("Evaluation results")
        st.stop()
    t1, t2, t3, t4 = st.tabs(["Robustness", "Confusion matrix", "Reliability", "External (PlantDoc)"])
    with t1:
        rob = read_csv("robustness_by_corruption.csv")
        d = rob[rob["family"] != "clean"]
        fig = px.bar(d, x="family", y="accuracy", color=d["severity"].astype(str), barmode="group",
                     labels={"color": "severity"}, color_discrete_sequence=["#9cc58a", LEAF, FOREST])
        fig.add_hline(y=clf["accuracy"], line_dash="dot", annotation_text=f"clean {clf['accuracy']:.1%}")
        fig.update_layout(height=380, yaxis_tickformat=".0%", title="Test accuracy by corruption type and severity")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(rob.round(4), hide_index=True, use_container_width=True)
    with t2:
        cm = pd.read_csv(METRICS / "test_confusion_matrix.csv", index_col=0)
        norm = cm.div(cm.sum(axis=1).clip(lower=1), axis=0)
        labels = [pretty(c) for c in cm.index]
        fig = go.Figure(go.Heatmap(z=norm.values, x=labels, y=labels, colorscale="Greens", zmin=0, zmax=1,
                                   hovertemplate="true %{y}<br>pred %{x}<br>%{z:.1%}<extra></extra>"))
        fig.update_layout(height=760, title=f"Row-normalised confusion matrix (clean test, n={clf['test_images']:,})",
                          xaxis=dict(tickfont=dict(size=8)), yaxis=dict(tickfont=dict(size=8), autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(read_csv("test_per_class_metrics.csv").round(4), use_container_width=True)
    with t3:
        if not rel:
            need_setup("Reliability results")
        else:
            cmp_ = read_csv("reliability_baseline_comparison.csv")
            st.markdown("Positive class = **incorrect** prediction. Thresholds chosen on reliability-validation data (maximise F2).")
            st.dataframe(cmp_.round(4), hide_index=True, use_container_width=True)
            long = cmp_.melt(id_vars="method", value_vars=["precision", "recall", "f1", "pr_auc"])
            st.plotly_chart(px.bar(long, x="variable", y="value", color="method", barmode="group", height=360,
                                   title="Reliability: Random Forest vs baselines (test)"), use_container_width=True)
            fig = go.Figure()
            for name, r in rel["methods"].items():
                fig.add_trace(go.Scatter(x=r["risk_coverage"]["coverage"], y=r["risk_coverage"]["error_rate"], name=name, mode="lines"))
            fig.update_layout(height=340, title="Risk vs coverage (lower is better)", xaxis_title="Share of predictions accepted",
                              yaxis_title="Error rate among accepted", yaxis_tickformat=".1%")
            st.plotly_chart(fig, use_container_width=True)
            imp = read_csv("reliability_feature_importance_B.csv")
            if imp is not None:
                st.plotly_chart(px.bar(imp.sort_values("permutation_importance_pr_auc"), x="permutation_importance_pr_auc", y="feature",
                                       orientation="h", height=460, title="Feature importance (RF-B, permutation, reliability-validation)"),
                                use_container_width=True)
            cal = pd.DataFrame(rel["calibration_B"]["bins"])
            fig = go.Figure([go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dot", color="#999"), name="perfect"),
                             go.Scatter(x=cal["mean_predicted"], y=cal["observed_incorrect"], mode="lines+markers", name="RF-B")])
            fig.update_layout(height=320, title=f"Calibration of the risk score (Brier {rel['calibration_B']['brier']:.4f})",
                              xaxis_title="Predicted risk", yaxis_title="Observed share incorrect")
            st.plotly_chart(fig, use_container_width=True)
    with t4:
        if not ext:
            need_setup("External evaluation")
        else:
            for key, title in (("strict_exact_mappings", "Strict (exact label matches)"), ("extended_with_ambiguous", "Extended (adds ambiguous matches)")):
                s = ext[key]
                st.markdown(f"**{title}:** {s['images']:,} images, {s['classes']} classes - accuracy {s['accuracy']:.1%}, "
                            f"top-3 {s['top3_accuracy']:.1%}, macro-F1 {s['macro_f1_over_present_classes']:.3f}")
            pc = pd.DataFrame(ext["per_class_accuracy_strict"]).sort_values("accuracy")
            st.plotly_chart(px.bar(pc, x="accuracy", y="plantdoc_class", orientation="h", height=620, hover_data=["images"],
                                   title="PlantDoc accuracy per class (strict)").update_layout(xaxis_tickformat=".0%"), use_container_width=True)
            st.caption(ext["note"])

else:  # About
    hero("About", "AgriShield", "A research demonstration of reliability-aware plant disease detection under image degradation.")
    st.markdown("""
**Motivation.** A disease classifier can be highly accurate on clean benchmark photos yet fail silently on dark, blurred, noisy or
compressed images taken in the field. AgriShield measures that failure and trains a second model to warn when a prediction is
likely to be wrong.

**EfficientNet-B0.** A compact convolutional network (about 4 million parameters) pre-trained on ImageNet and fine-tuned here on
PlantVillage leaf photos (transfer learning).

**Random Forest.** An ensemble of decision trees trained on a separate calibration split. It predicts *incorrect (1) vs correct (0)*
from the classifier's confidence, margin and entropy, image-quality indicators and (model B) how stable the prediction is under
six fixed, mild perturbations.

**Data.** PlantVillage (Mohanty, Hughes & Salathé, 2016; github.com/spMohanty/PlantVillage-Dataset; CC BY-SA 3.0) for
training and internal testing. PlantDoc (Singh et al., 2020; github.com/pratikkayal/PlantDoc-Dataset; CC BY 4.0) for external
testing only. Software: PyTorch/torchvision, scikit-learn, Streamlit, Plotly.

**Process.** Duplicate-aware split → train/validate EfficientNet → compute features on calibration images → tune the Random Forest and
thresholds on reliability-validation data → evaluate everything once on the locked test set → external test on PlantDoc.

**Contribution.** An engineering and evaluation study: combining a standard classifier with a learned error detector and comparing it
with confidence and margin baselines under controlled degradations. It does not claim a new algorithm.

**Limitations.** PlantVillage photos are lab-style (single leaf, plain background), so external accuracy on field photos is much lower.
The reliability model learned from PlantVillage-style errors and synthetic degradations; its risk score is not a guarantee.

**Disclaimer.** This is a research demonstration, not a replacement for diagnosis by a qualified agricultural expert.
See LICENSES_AND_ATTRIBUTION.md for dataset and software licences.
""")
    if data_rep:
        st.caption(f"Dataset check: {data_rep['plantvillage']['files']:,} PlantVillage files, {data_rep['plantdoc']['files']:,} PlantDoc files.")
