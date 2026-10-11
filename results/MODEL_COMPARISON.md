# AgriShield — EfficientNet-B0 vs Random Forest (verified from `results/metrics.json` and `reports/metrics/`)

## 1. EfficientNet-B0 in one line
It looks at a leaf photo and says **which disease it is** (38 classes).

## 2. Random Forest in one line
It looks at *how the prediction was made* (confidence, image quality, stability) and says **whether that disease prediction can be trusted**.

## 3. Main comparison table
| Parameter | EfficientNet-B0 | Random Forest (final = config B) |
|---|---|---|
| Model type | Convolutional neural network (torchvision, 4.06 M parameters) | Ensemble of 400 decision trees (scikit-learn) |
| ML category | Supervised deep learning, multi-class classification | Supervised classical ML, binary classification |
| Main purpose | Identify the plant disease | Predict that EfficientNet's answer is **wrong** |
| Input | Leaf image (RGB) | 17 numbers computed from the image and from EfficientNet's output |
| Output | Disease class + confidence | Risk score P(incorrect) → "reliable" or "warning" (threshold 0.1175) |
| Role in AgriShield | Primary predictor | Safety / reliability layer |
| Features used | Raw image pixels (learned) | 4 confidence features (max prob, margin, entropy, top-3 mass), 7 image-quality features (brightness, contrast, sharpness, noise, saturation, image size, aspect ratio), 6 stability features (agreement of the prediction under 6 mild perturbations) |
| Training data | PlantVillage train split: 38,014 images (val 5,433; test 5,432) | Calibration split: 3,801 images (`rel_train`) × 4 conditions (clean + 3 random corruptions); 1,625 images (`rel_val`) for tuning + thresholds |
| Number of classes | 38 diseases/healthy | 2 (correct / incorrect) |
| Main metric | Accuracy, macro-F1 | PR-AUC (positive = incorrect prediction) |
| Actual result | Accuracy 99.80%, macro-F1 0.9961 | PR-AUC 0.947 |
| Secondary metrics | Macro precision 0.9966, macro recall 0.9957; PlantDoc 18.0% | Recall 97.4%, precision 66.7%, F1 0.792, ROC-AUC 0.986, Brier 0.0347 |
| Strength | Near-perfect on clean lab images | Catches 97.4% of wrong predictions; beats confidence alone |
| Limitation | Collapses under severe blur/noise/JPEG/low-res; only 18.0% on field photos | Precision 66.7% (about 1 false alarm per 2 true alarms); learned on PlantVillage-style errors, ROC-AUC only 0.625 on PlantDoc |

## 4. Performance comparison (only metrics valid for each task)
**EfficientNet-B0 — PlantVillage clean test set, 5,432 images**
| Metric | Value |
|---|---|
| Accuracy | 99.80% |
| Macro precision / recall / F1 | 0.9966 / 0.9957 / 0.9961 |
| External PlantDoc (2,234 images, 25 classes) | accuracy 18.04%, top-3 38.36%, macro-F1 0.176 |

Robustness (accuracy at the most severe level; drop vs clean 99.80%):
| Degradation | Accuracy | Drop |
|---|---|---|
| Noise (σ=0.2) | 2.15% | 97.6 pts |
| Low resolution (0.125×) | 29.51% | 70.3 pts |
| JPEG (quality 5) | 41.37% | 58.4 pts |
| Blur (σ=3) | 52.54% | 47.3 pts |
| Brightness bright / dark | 91.62% / 94.33% | 8.2 / 5.5 pts |

**Random Forest reliability — 21,728 test rows (5,432 images × 4 conditions), 17.9% of predictions incorrect**
| Metric | RF (B) | RF (A, no extra passes) | Confidence baseline | Margin baseline |
|---|---|---|---|---|
| PR-AUC | **0.947** | 0.886 | 0.848 | 0.809 |
| ROC-AUC | 0.986 | 0.972 | 0.962 | 0.959 |
| Recall (errors caught) | 97.4% | 95.0% | 95.7% | 95.9% |
| Precision | 66.7% | 63.7% | 57.7% | 56.9% |
| F1 | 0.792 | 0.763 | 0.720 | 0.714 |
| Brier score | 0.0347 | 0.0559 | – | – |

Random guessing would give PR-AUC ≈ 0.179 (the error rate).

## 5. Why accuracy cannot compare the two
- EfficientNet answers *"What disease is this?"* (38-way). Random Forest answers *"Is that answer wrong?"* (yes/no).
- Their targets differ: 99.80% is the share of leaves labelled correctly; the RF's target is itself the **rare** event "EfficientNet is wrong" (0.2% on clean images, 17.9% on this mixed test). Accuracy on a rare event is misleading — always saying "reliable" would score 82% accuracy and catch nothing. That is why the RF uses **PR-AUC, recall and Brier**.
- Even the datasets differ: EfficientNet is scored on 5,432 clean images; the RF is scored on 21,728 clean + degraded rows.
- Better framing from the project: the RF is judged **against EfficientNet's own confidence**, the only fair rival for the same question.

## 6. How the two models work together
```
IMAGE
  ↓
EfficientNet-B0  →  disease + confidence
  ↓
6 mild perturbations (dark, bright, blur, noise, JPEG 60, low-res 0.6) re-run through EfficientNet → stability features
  ↓
Feature extraction (confidence + image quality + stability = 17 features)
  ↓
Random Forest  →  risk P(incorrect)
  ↓
risk ≥ 0.1175 → WARNING   |   risk < 0.1175 → TRUSTWORTHY
```
- **EfficientNet alone:** gives a confident answer even when wrong. On degraded rows, errors occur in 23.8% of cases. The `confidence_vs_correctness.png` graph shows even 0.95–0.99 confidence is wrong 19% of the time.
- **Problem solved:** knowing *when not to trust* the classifier.
- **Why necessary:** confidence alone gives PR-AUC 0.848; adding image quality and stability features raises it to 0.947 (and Brier 0.035 means the risk numbers are meaningful probabilities).
- **Improvement to the system:** it turns a silent wrong answer into a warning; 97.4% of wrong predictions are flagged (at the cost of flagging 26% of all predictions).

## 7. Which model is better?
Neither — they do different jobs. EfficientNet-B0 is better for **disease classification**; Random Forest is better for **reliability prediction**. They are complementary: the strength of AgriShield is the combination.

## 8. 30-second explanation
"AgriShield uses two models. EfficientNet-B0 identifies the disease and is 99.8% accurate on clean images, but its accuracy collapses when images are blurred, noisy or compressed — and it stays confident. A Random Forest watches EfficientNet's behaviour and image quality and predicts whether the answer is wrong; it catches 97.4% of the errors with PR-AUC 0.947 against 0.848 for plain confidence. So EfficientNet gives the answer and the Random Forest tells you whether to trust it."

## 9. 2-minute explanation
Plant disease models trained on clean lab photos reach almost perfect accuracy — ours gets 99.80% and macro-F1 0.996 on PlantVillage. But farmers' photos are blurred, dark or compressed. When we degrade the test images, accuracy falls to 52.5% (blur), 41.4% (JPEG), 29.5% (low resolution) and 2.2% (noise) at the highest severity, and on the external PlantDoc field dataset it is only 18%. The dangerous part is that the network is often still confident.

So we add a second model with a different job. For each prediction we compute 17 numbers: how confident EfficientNet is, how good the image is (brightness, sharpness, noise…), and how stable the prediction stays when we slightly perturb the image. A Random Forest turns these into a risk score for "this prediction is wrong". Because wrong predictions are rare, we evaluate it with PR-AUC, recall and Brier score rather than accuracy. It reaches PR-AUC 0.947 versus 0.848 for the confidence baseline, recall 97.4%, precision 66.7% and Brier 0.035.

The two models are not competitors: one answers "what disease?" and the other answers "can we trust that?". Together they give a prediction plus a trustworthy/warning decision. Honest limit: the RF is trained on PlantVillage-style errors, so on PlantDoc its separation is weaker (ROC-AUC 0.625).

## 10. Viva questions and answers
1. **Why two models?** The disease classifier cannot judge its own mistakes; a second model is trained specifically to predict them.
2. **Why EfficientNet-B0?** Small (4M parameters), fast (~12 ms/image), pretrained on ImageNet, and 99.8% on PlantVillage — suitable for a deployable app.
3. **Why Random Forest?** Our reliability inputs are 17 small tabular numbers; forests handle tabular data well, need little tuning, and give probabilities.
4. **Why not RF for classification?** It would need hand-made features from images; it can't learn visual patterns as a CNN does.
5. **Why not EfficientNet alone?** Because it stays confident when wrong (e.g. 19% of its 0.95–0.99-confidence predictions were wrong), so users get no warning.
6. **Which has higher accuracy?** Question is invalid: 99.80% is disease accuracy; the RF has a different target, so we report PR-AUC/recall instead.
7. **Can you compare their accuracy directly?** No — different questions, different datasets and class balance.
8. **Input to RF?** 17 numbers: 4 confidence, 7 image-quality, 6 stability features.
9. **What does RF predict?** The probability that EfficientNet's prediction is incorrect.
10. **How does RF know EfficientNet may be wrong?** Low margin/high entropy, poor image quality, or the prediction changing under small perturbations are patterns that, in training data, went with wrong answers.
11. **PR-AUC?** Area under the precision–recall curve: how well the model ranks the true errors first, which matters when errors are rare.
12. **Why PR-AUC?** Only 17.9% of rows are errors; ROC-AUC and accuracy look good even for weak models on imbalanced data.
13. **Confidence baseline?** Flag a prediction as risky when EfficientNet's max probability is low — the obvious no-extra-model method.
14. **How does it improve the system?** It converts silent errors into warnings: 97.4% of wrong predictions are flagged; PR-AUC 0.947 vs 0.848.
15. **Blurred/noisy input?** EfficientNet accuracy drops sharply (blur sev 3: 52.5%), image-quality and stability features change, RF risk rises, and the app shows a warning.
16. **EfficientNet wrong?** If the RF risk ≥ 0.1175 the user sees a warning instead of a trusted answer; 2.6% of errors still slip through.
17. **Novelty?** Not a new algorithm — we claim an engineering/evaluation framework: leakage-aware splits, a degradation benchmark, and a learned error detector shown to beat confidence. No novelty beyond that is claimed (repo report says no literature review was done).

## Verified-vs-previous numbers
All previous numbers match. One wording fix: "noise 2.2%, low-res 29.5%, JPEG 41.4%, blur 52.5%" are **accuracies** at the most severe level, not drops. Actual drops: noise 97.6, low-res 70.3, JPEG 58.4, blur 47.3 points.
Caveat: on clean rows only, RF(B) PR-AUC is 0.320 vs confidence 0.414 (only 11 clean errors, unstable) — the RF's advantage comes from degraded images.
