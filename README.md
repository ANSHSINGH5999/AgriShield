# AgriShield

**Reliability-Aware Plant Disease Detection Under Image Quality Degradation.** This is a research demonstration, not a
replacement for diagnosis by an agricultural expert.

AgriShield has two parts:

1. **EfficientNet-B0** recognises one of 38 plant/disease classes in a leaf photo (trained on PlantVillage).
2. **Random Forest** estimates the risk that this prediction is **wrong**. It uses the classifier's confidence, image-quality
   indicators and, optionally, how stable the prediction stays under 6 small fixed changes to the image.

A Streamlit website shows the measured results and lets you test your own images.

- **Results website:** https://agrishield.vercel.app (measured results, no installation needed)
- **Code, trained models and the interactive app:** https://github.com/ANSHSINGH5999/AgriShield

## Measured results (from `reports/metrics`, full details in `reports/TECHNICAL_REPORT.md`)

**Clean PlantVillage test set** (5,432 held-out images, 38 classes): accuracy **99.80%**,
macro-F1 **0.9961**.

**Accuracy at the strongest severity of each degradation** (same test images):

| Degradation (severity 3) | Accuracy |
|---|---:|
| gaussian noise | 2.2% |
| low resolution | 29.5% |
| jpeg compression | 41.4% |
| gaussian blur | 52.5% |
| brightness high | 91.6% |
| brightness low | 94.3% |

**Reliability.** Positive = incorrect prediction. Test rows are clean + 3 sampled degradations per image, and thresholds were chosen on
validation data.

| Method | PR-AUC | Precision | Recall | False-alarm rate |
|---|---:|---:|---:|---:|
| Confidence threshold | 0.848 | 57.7% | 95.7% | 15.3% |
| Margin threshold | 0.809 | 56.9% | 95.9% | 15.9% |
| Random Forest (A) | 0.886 | 63.7% | 95.0% | 11.8% |
| Random Forest (B) | 0.947 | 66.7% | 97.4% | 10.6% |

**External PlantDoc field photos** (2,234 images, exact label matches): accuracy **18.0%**, top-3 38.4%.
The model trained on lab-style PlantVillage photos does **not** transfer well to field photos. On clean images alone, the confidence
baseline beat the Random Forest. The Random Forest's advantage comes from degraded images. See the technical report.


---

## Quick start (Windows) — no training needed

The trained models are already in the `models` folder.

1. **Install Python 3.12** from <https://www.python.org/downloads/>. During installation tick **"Add python.exe to PATH"**.
   Check it: open *Command Prompt* and run `py -3.12 --version`. It should print `Python 3.12.x`.
2. **Open the project folder.** Copy `AgriShield` anywhere, for example `C:\Users\you\AgriShield`, and open it in File Explorer.
3. **Double-click `setup_windows.bat`** (run it once). It creates a virtual environment in `.venv`, installs the pinned packages
   (about 1–3 GB, mostly PyTorch) and runs the tests.
4. **Double-click `run_app.bat`.**
5. Open the **Local URL** it prints, usually <http://localhost:8501>.

### Doing steps 3–4 by hand in Command Prompt
```bat
cd C:\path\to\AgriShield
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m pytest -q tests
streamlit run app.py
```
On macOS or Linux, use `python3.12 -m venv .venv` and `source .venv/bin/activate`.

---

## Using the website

| Page | What it shows |
|---|---|
| **Dashboard** | Measured test accuracy, macro-F1, reliability metrics and external (PlantDoc) results, each labelled |
| **Disease Detection** | Upload a JPG/PNG to see the predicted disease, top-3, confidence, reliability risk and quality indicators |
| **Robustness Lab** | Change brightness, blur, noise, JPEG quality and resolution; compares the original and altered predictions |
| **Reliability Report** | Risk, status, threshold (adjustable; default chosen on validation data), stability and observations |
| **Model Evaluation** | Robustness by corruption, confusion matrix, reliability vs baselines, feature importance, external results |
| **About** | Method, data sources, limitations and disclaimer |

Sample images (from the PlantVillage **test** split) are in `assets/samples` and can be picked in the app.

**Four different numbers, kept separate:**
- **Test accuracy** is measured once on held-out labelled test images.
- **Prediction confidence** is the model's probability for its predicted class on *your* image. It is not accuracy.
- **Reliability risk** is the Random Forest's estimated chance that the prediction is wrong. It is a score, not a guarantee.
- **Stability** is the share of the 6 slightly-changed copies of your image that keep the same prediction.

---

## Project structure
```
AgriShield/
├── app.py                      Streamlit website
├── config.yaml                 every setting (seed, split, training, perturbations, stability policy)
├── src/                        library code
│   ├── imaging.py              loading, standardising (shorter side 256), preprocessing (crop 224, ImageNet normalisation)
│   ├── perturbations.py        brightness, blur, noise, JPEG, low-resolution
│   ├── quality.py              brightness, contrast, sharpness, noise estimate, saturation, size
│   ├── reliability.py          reliability features, fixed feature schema, Random Forest loader
│   ├── predictor.py            image -> prediction + top-3 + quality + stability + risk (used by the app)
│   ├── model.py, data.py, datasets.py, engine.py, external_mapping.py, reliability_metrics.py, ui.py, plots.py
├── scripts/                    pipeline steps (see "Training")
├── models/                     trained files (see below)
├── reports/                    metrics (JSON/CSV), figures (PNG), manifests, TECHNICAL_REPORT.md
├── assets/samples/             a few test images for the demo
├── tests/                      automated tests (pytest)
├── setup_windows.bat, run_app.bat, train_windows.bat, evaluate_windows.bat
├── requirements.txt            pinned versions
└── LICENSES_AND_ATTRIBUTION.md
```

### Model files (`models/`)
| File | Content |
|---|---|
| `efficientnet_b0.pt` | classifier weights (state dict, loaded with `weights_only=True`) |
| `class_names.json` | class index → class name (the exact training order) |
| `preprocessing.json` | resize, crop, RGB order and normalisation |
| `model_metadata.json` | architecture, best epoch, seed, versions |
| `reliability_rf_A.joblib`, `reliability_rf_B.joblib` | Random Forests (A: image only; B: + stability, 6 extra passes) |
| `reliability_schema.json` | feature order, thresholds, hyper-parameters, positive-class definition |
| `reliability_threshold.json` | validation-selected warning thresholds |
| `quality_reference.json` | typical ranges of the quality indicators (used for explanations) |

The `.joblib` files use Python pickle. Only load model files that come from this project.

---

## Training (optional)

You only need this to reproduce or change the models.

**Requirements.** About 3 GB of disk for the datasets and 8 GB RAM. With an NVIDIA GPU or an Apple M-series chip, the whole pipeline
takes a few hours. On a CPU-only laptop, expect a day or more. Training does not need an internet connection, except for downloading
the datasets and the ImageNet starting weights (20 MB, downloaded automatically once).

1. **Download the datasets** into a `data` folder inside AgriShield:
   ```bat
   mkdir data
   cd data
   git clone --depth 1 --filter=blob:none --sparse https://github.com/spMohanty/PlantVillage-Dataset.git
   cd PlantVillage-Dataset
   git sparse-checkout set raw/color
   cd ..
   git clone --depth 1 https://github.com/pratikkayal/PlantDoc-Dataset.git
   cd ..
   ```
   The expected layout is `data\PlantVillage-Dataset\raw\color\<38 class folders>` and `data\PlantDoc-Dataset\train|test\<classes>`.
2. **Train:** double-click `train_windows.bat`, or run `python -m scripts.run_pipeline --train`. This runs, in order:
   - `prepare_data`: inspect the data, remove duplicates and build group splits → `reports/manifests`
   - `train_classifier`: train EfficientNet-B0 and pick the epoch on validation → `models/efficientnet_b0.pt`
   - `compute_features --split calibration`: build the reliability training data
   - `train_reliability`: fit the Random Forest, with hyper-parameters and thresholds chosen on reliability-validation
3. **Evaluate:** double-click `evaluate_windows.bat`, or run `python -m scripts.run_pipeline --evaluate`. This runs, in order:
   - `compute_features --split test`: the locked test set
   - `evaluate_classifier`: clean metrics and robustness
   - `evaluate_reliability`: Random Forest vs baselines
   - `evaluate_external`: PlantDoc
   - `make_report`: `reports/TECHNICAL_REPORT.md`

To check the code quickly before a long run: `python -m scripts.train_classifier --epochs 1 --limit 512`.
On Windows, if data loading hangs, set `num_workers: 0` in `config.yaml`.

### Leakage safeguards
- Exact and near-duplicate images are grouped, and each group goes to exactly one partition.
- Partitions: train (fit EfficientNet), val (choose the epoch), calibration (Random Forest: `rel_train` fits it, `rel_val` picks
  hyper-parameters and thresholds), test (used only in the evaluation steps).
- The Random Forest never sees the true label as a feature. Its features come from predictions on images EfficientNet was not
  trained on.
- PlantDoc is never used for training. PlantDoc files identical to PlantVillage files are excluded.

---

## Troubleshooting
| Problem | Fix |
|---|---|
| `'py' is not recognized` | Reinstall Python and tick "Add python.exe to PATH", or replace `py -3.12` with `python` |
| `pip install` fails on torch | Use 64-bit Python 3.12 (not 3.13 or 32-bit), then run `python -m pip install --upgrade pip` and try again |
| `Missing model file: models/...` | The `models` folder is incomplete: copy it from the original AgriShield folder, or retrain |
| Port 8501 already in use | Run `streamlit run app.py --server.port 8502` |
| Website shows "not available yet" | `reports/metrics` is missing: copy it, or run `evaluate_windows.bat` |
| Training data loading hangs on Windows | Set `training: num_workers: 0` in `config.yaml` |
| Want GPU training on Windows | Install the CUDA build of PyTorch from <https://pytorch.org/get-started/locally/>; the code uses CUDA automatically |
| Antivirus blocks `.bat` files | Run the commands from "Doing steps 3–4 by hand" |

## Reproducibility
- Seed `42` is used for splits, training and the Random Forest (`config.yaml`).
- Split manifests are in `reports/manifests`, and training history is in `reports/metrics/training_history.json`.
- Model selection is in `reliability_model_selection.json`, with all metrics in `reports/metrics`.
- GPU training (CUDA/MPS) is not bit-for-bit deterministic, so retraining can give slightly different numbers.
