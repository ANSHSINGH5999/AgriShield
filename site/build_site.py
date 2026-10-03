"""Builds the static results website (site/public/) from reports/ - every number is read from the saved files.

    python site/build_site.py
"""
import html
import json
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
M, F = ROOT / "reports" / "metrics", ROOT / "reports" / "figures"
OUT = ROOT / "site" / "public"
REPO = "https://github.com/ANSHSINGH5999/AgriShield"


def j(name):
    return json.loads((M / name).read_text())


def pc(x, d=1):
    return f"{x * 100:.{d}f}%"


def table(df: pd.DataFrame) -> str:
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    rows = "".join("<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in r) + "</tr>" for r in df.itertuples(index=False))
    return f'<div class="tw"><table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table></div>'


def main():
    clf, rel, ext, ds = j("classifier_test_metrics.json"), j("reliability_test_metrics.json"), j("external_plantdoc_metrics.json"), j("dataset_report.json")
    hist = j("training_history.json")
    rob = pd.read_csv(M / "robustness_by_corruption.csv")
    cmp_ = pd.read_csv(M / "reliability_baseline_comparison.csv")
    imp = pd.read_csv(M / "reliability_feature_importance_B.csv")
    s, x = ext["strict_exact_mappings"], ext["extended_with_ambiguous"]
    pv = ds["plantvillage"]
    m = rel["methods"]
    rb, cb = m["Random Forest (B)"]["all conditions"], m["Confidence threshold"]["all conditions"]
    clean_cmp = {n: v["clean only"]["pr_auc"] for n, v in m.items()}

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)
    for fig in ["robustness_accuracy.png", "reliability_risk_coverage_calibration.png", "reliability_feature_importance.png",
                "confusion_matrix_test.png", "training_curves.png"]:
        shutil.copy(F / fig, OUT / "figures" / fig)
    samples = pd.read_csv(ROOT / "assets" / "samples" / "SAMPLES.csv")
    (OUT / "samples").mkdir(exist_ok=True)
    for name in samples["sample"]:
        shutil.copy(ROOT / "assets" / "samples" / name, OUT / "samples" / name)

    r = rob[rob["family"] != "clean"].copy()
    rob_tbl = r.assign(**{"degradation": r["family"].str.replace("_", " "), "accuracy": r["accuracy"].map(pc),
                          "macro-F1": r["macro_f1"].map(lambda v: f"{v:.3f}"),
                          "predictions changed": r["prediction_change_rate"].map(pc)})[
        ["degradation", "severity", "value", "accuracy", "macro-F1", "predictions changed"]]
    rel_tbl = cmp_.assign(**{"PR-AUC": cmp_["pr_auc"].map(lambda v: f"{v:.3f}"), "ROC-AUC": cmp_["roc_auc"].map(lambda v: f"{v:.3f}"),
                             "precision": cmp_["precision"].map(pc), "recall": cmp_["recall"].map(pc),
                             "false alarms": cmp_["false_alarm_rate"].map(pc)})[
        ["method", "PR-AUC", "ROC-AUC", "precision", "recall", "false alarms"]]
    worst = r.sort_values("accuracy").iloc[0]
    top_feats = ", ".join(imp["feature"].head(4).str.replace("_", " "))
    gallery = "".join(f'<figure><img src="samples/{html.escape(n)}" alt="{html.escape(c)}" loading="lazy">'
                      f'<figcaption>{html.escape(c.replace("___", " — ").replace("_", " "))}</figcaption></figure>'
                      for n, c in zip(samples["sample"], samples["true_class"]))

    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>AgriShield</title>
<meta name="description" content="Reliability-aware plant disease detection under image quality degradation - measured results.">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700&family=DM+Sans:wght@400;500;700&display=swap" rel="stylesheet">
<style>
:root{{--forest:#17432b;--leaf:#3f8a55;--sprout:#cfe8c4;--cream:#f7f6f0;--card:#fff;--ink:#1d2a22;--muted:#5d6b62;--line:#e3e6dc;--amber:#c98a1b;--red:#b5452f}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--cream);color:var(--ink);font:16px/1.6 'DM Sans','Segoe UI',system-ui,sans-serif}}
.wrap{{max-width:1100px;margin:0 auto;padding-inline:20px;padding-block:0 80px}}
header{{background:linear-gradient(135deg,#17432b,#24603d);color:#f3f8f1}}
header .wrap{{padding-block:56px 44px}}
.eyebrow{{text-transform:uppercase;letter-spacing:.12em;font-size:.75rem;font-weight:700;color:var(--sprout)}}
h1{{font:700 clamp(2.2rem,5vw,3.4rem)/1.05 Fraunces,Georgia,serif;margin:.3em 0 .35em;text-wrap:balance}}
header p{{max-width:66ch;color:#d7ead2;font-size:1.08rem;margin:0}}
.links{{display:flex;flex-wrap:wrap;gap:10px;margin-top:22px}}
.btn{{display:inline-block;padding:.6rem 1.1rem;border-radius:12px;font-weight:700;text-decoration:none}}
.btn.p{{background:var(--sprout);color:var(--forest)}}.btn.s{{border:1px solid rgba(255,255,255,.4);color:#fff}}
h2{{font:700 1.9rem/1.15 Fraunces,Georgia,serif;color:var(--forest);margin:0 0 .3em;text-wrap:balance}}
section{{padding-top:52px}}.sub{{color:var(--muted);max-width:72ch;margin:0 0 18px}}
.tag{{display:inline-block;background:var(--sprout);color:var(--forest);border-radius:999px;padding:3px 11px;font-size:.75rem;font-weight:700;margin:18px 0 10px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:14px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:18px;padding:18px 20px}}
.card .l{{font-size:.76rem;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);font-weight:700}}
.card .v{{font:600 2rem/1.15 Fraunces,Georgia,serif;color:var(--forest);margin-top:4px;font-variant-numeric:tabular-nums}}
.card .n{{font-size:.85rem;color:var(--muted)}}
.tw{{overflow-x:auto;background:var(--card);border:1px solid var(--line);border-radius:16px}}
table{{border-collapse:collapse;width:100%;font-size:.93rem}}th,td{{padding:9px 14px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap;font-variant-numeric:tabular-nums}}
th{{font-size:.76rem;text-transform:uppercase;letter-spacing:.06em;color:var(--muted)}}tr:last-child td{{border-bottom:0}}
img.fig{{width:100%;height:auto;background:#fff;border:1px solid var(--line);border-radius:16px;margin-top:14px}}
.two{{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:18px}}
.note{{border-radius:14px;padding:14px 18px;margin-top:16px;max-width:85ch}}
.note.w{{background:#fdf3e1;border:1px solid #efd29a;color:#6b4a0c}}.note.ok{{background:#e8f4e3;border:1px solid #b9dcae;color:#1d4d2c}}
.gallery{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}}
.gallery figure{{margin:0;background:var(--card);border:1px solid var(--line);border-radius:14px;padding:8px}}
.gallery img{{width:100%;height:auto;border-radius:10px;display:block}}.gallery figcaption{{font-size:.8rem;color:var(--muted);margin-top:6px}}
ol,ul{{max-width:80ch;padding-left:22px}}li{{margin:5px 0}}
pre{{background:#10241a;color:#e3f1df;border-radius:14px;padding:16px 18px;overflow-x:auto;font-size:.88rem}}
footer{{color:var(--muted);font-size:.85rem;padding-top:56px}}
a{{color:var(--leaf)}}a:focus-visible,.btn:focus-visible{{outline:3px solid var(--amber);outline-offset:2px}}
</style></head><body>
<header><div class="wrap">
<div class="eyebrow">Research demonstration · measured results</div>
<h1>AgriShield</h1>
<p>Reliability-aware plant disease detection under image quality degradation. EfficientNet-B0 recognises 38 leaf conditions.
A Random Forest estimates the risk that each prediction is wrong.</p>
<div class="links"><a class="btn p" href="{REPO}">Code, models &amp; app on GitHub</a><a class="btn s" href="#run">Run the live app</a></div>
</div></header>
<div class="wrap">

<section>
<h2>Results at a glance</h2>
<p class="sub">Every number on this page is read from the project's saved evaluation files. Internal and external results are kept separate.</p>
<span class="tag">INTERNAL TEST · PlantVillage, locked held-out set</span>
<div class="grid">
<div class="card"><div class="l">Test accuracy</div><div class="v">{pc(clf['accuracy'], 2)}</div><div class="n">clean images</div></div>
<div class="card"><div class="l">Macro F1-score</div><div class="v">{clf['macro_f1']:.3f}</div><div class="n">{clf['num_classes']} classes</div></div>
<div class="card"><div class="l">Test images</div><div class="v">{clf['test_images']:,}</div><div class="n">never used for training or tuning</div></div>
<div class="card"><div class="l">Model size</div><div class="v">{clf['model_file_mb']:.1f} MB</div><div class="n">{clf['parameters']:,} parameters</div></div>
</div>
<span class="tag">RELIABILITY · positive = incorrect prediction</span>
<div class="grid">
<div class="card"><div class="l">PR-AUC, Random Forest B</div><div class="v">{rb['pr_auc']:.3f}</div><div class="n">confidence baseline {cb['pr_auc']:.3f}</div></div>
<div class="card"><div class="l">Recall of errors</div><div class="v">{pc(rb['recall'])}</div><div class="n">baseline {pc(cb['recall'])}</div></div>
<div class="card"><div class="l">Precision of warnings</div><div class="v">{pc(rb['precision'])}</div><div class="n">baseline {pc(cb['precision'])}</div></div>
<div class="card"><div class="l">False alarms</div><div class="v">{pc(rb['false_alarm_rate'])}</div><div class="n">baseline {pc(cb['false_alarm_rate'])}</div></div>
</div>
<span class="tag">EXTERNAL · PlantDoc field photos (never trained on)</span>
<div class="grid">
<div class="card"><div class="l">External accuracy</div><div class="v">{pc(s['accuracy'])}</div><div class="n">{s['images']:,} images, exact label matches</div></div>
<div class="card"><div class="l">External top-3 accuracy</div><div class="v">{pc(s['top3_accuracy'])}</div><div class="n">with ambiguous matches: {pc(x['accuracy'])} top-1</div></div>
</div>
<div class="note w"><b>Read this before the 99.8%.</b> PlantVillage photos show one leaf on a plain background. On real field photos (PlantDoc)
the same model reached {pc(s['accuracy'])}. High accuracy on the benchmark does not mean the model works in the field.</div>
</section>

<section>
<h2>How much do bad photos hurt?</h2>
<p class="sub">The same {clf['test_images']:,} test images were degraded at three severities. The largest drop was
{html.escape(worst['family'].replace('_', ' '))} at severity {int(worst['severity'])}: accuracy {pc(worst['accuracy'])}.</p>
<img class="fig" src="figures/robustness_accuracy.png" alt="Test accuracy and prediction-change rate by degradation type and severity">
{table(rob_tbl)}
</section>

<section>
<h2>Can we tell when the model is wrong?</h2>
<p class="sub">The Random Forest was trained only on a separate calibration split. Hyper-parameters and every threshold were chosen on
reliability-validation data. Test rows are each image clean plus 3 sampled degradations ({rel['test_rows']:,} rows).
Configuration A uses only the uploaded image. Configuration B adds stability under 6 fixed mild perturbations (6 extra model passes).</p>
{table(rel_tbl)}
<div class="two"><img class="fig" src="figures/reliability_risk_coverage_calibration.png" alt="Risk-coverage curves and calibration of the reliability score">
<img class="fig" src="figures/reliability_feature_importance.png" alt="Permutation importance of reliability features"></div>
<p class="sub" style="margin-top:14px">The most useful signals were {html.escape(top_feats)}.</p>
<div class="note w"><b>Honest caveat.</b> On clean images alone, errors were rare. The simple confidence threshold scored a higher PR-AUC
({clean_cmp['Confidence threshold']:.3f}) than Random Forest B ({clean_cmp['Random Forest (B)']:.3f}). The Random Forest's advantage comes
from degraded images. On PlantDoc, its ranking ability was weak (ROC-AUC {s['reliability_B']['roc_auc']:.2f}).</div>
</section>

<section>
<h2>Training and test details</h2>
<div class="two"><img class="fig" src="figures/training_curves.png" alt="Training and validation curves">
<img class="fig" src="figures/confusion_matrix_test.png" alt="Confusion matrix on the clean test set"></div>
<ul>
<li><b>Data:</b> PlantVillage, {pv['files']:,} images in {pv['classes']} classes. The split is by duplicate group (exact copies, pixel-confirmed
near-duplicates and photos of the same leaf stay together): {', '.join(f"{k} {v:,}" for k, v in pv['split_sizes'].items())}. Seed {pv['split_seed']}.</li>
<li><b>Classifier:</b> EfficientNet-B0 (ImageNet start), fine-tuned for {len(hist['history'])} epochs. The best epoch was chosen on validation macro-F1.</li>
<li><b>Leakage safeguards:</b> the scaler and thresholds come from non-test data, the test set was evaluated once, and PlantDoc was never trained on.</li>
</ul>
</section>

<section>
<h2>Sample test images</h2>
<p class="sub">These are bundled with the app so it works without downloading the datasets (PlantVillage, CC BY-SA 3.0).</p>
<div class="gallery">{gallery}</div>
</section>

<section id="run">
<h2>Run the live app</h2>
<p class="sub">The interactive app runs on your own computer: Disease Detection, Robustness Lab, Reliability Report and Model Evaluation.
It loads the trained models, so no training is needed.</p>
<pre>git clone {REPO}.git
cd AgriShield
# Windows: double-click setup_windows.bat once, then run_app.bat
python -m venv .venv &amp;&amp; .venv\\Scripts\\activate      (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
streamlit run app.py</pre>
</section>

<footer>AgriShield is a research demonstration, not a replacement for diagnosis by an agricultural expert. Data: PlantVillage
(Mohanty, Hughes &amp; Salathé, 2016; CC BY-SA 3.0) and PlantDoc (Singh et al., 2020; CC BY 4.0). No novelty is claimed.
Source: <a href="{REPO}">{REPO}</a></footer>
</div></body></html>"""
    (OUT / "index.html").write_text(page, encoding="utf-8")
    (ROOT / "site" / "vercel.json").write_text(json.dumps({"outputDirectory": "public", "cleanUrls": True,
        "headers": [{"source": "/(.*)", "headers": [{"key": "Cache-Control", "value": "public, max-age=0, must-revalidate"}]}]}, indent=2))
    print(f"built {OUT / 'index.html'}")


if __name__ == "__main__":
    main()
