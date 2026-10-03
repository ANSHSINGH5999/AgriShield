# Licences and attribution

## Datasets (not bundled, except a few sample images)

| Dataset | Source and version used | Licence (as stated by the source) | Use in AgriShield |
|---|---|---|---|
| PlantVillage (raw/color) | https://github.com/spMohanty/PlantVillage-Dataset, commit `7f7ecc7e1eaca78107e3affe7cb5abd9427e139a` | CC BY-SA 3.0 (repository `README_HF.md`) | training, validation, calibration, test |
| PlantDoc | https://github.com/pratikkayal/PlantDoc-Dataset, commit `5467f6012d78d1c446145d5f582da6096f852ae8` | CC BY 4.0 (repository `LICENSE.txt`) | external evaluation only, never trained on |

The sample images in `assets/samples/` are unmodified PlantVillage **test-split** images, redistributed under CC BY-SA 3.0
with attribution to the PlantVillage authors.

**Citations**
- Mohanty, S. P., Hughes, D. P., & Salathé, M. (2016). *Using deep learning for image-based plant disease detection.*
  Frontiers in Plant Science, 7. https://doi.org/10.3389/fpls.2016.01419
- Singh, D., Jain, N., Jain, P., Kayal, P., Kumawat, S., & Batra, N. (2020). *PlantDoc: A Dataset for Visual Plant Disease
  Detection.* Proceedings of the 7th ACM IKDD CoDS and 25th COMAD, 249–253. https://doi.org/10.1145/3371158.3371196

## Model
- **EfficientNet-B0** architecture: Tan, M., & Le, Q. (2019). *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.*
  ICML. Implementation and ImageNet weights come from **torchvision** (BSD-3-Clause; the ImageNet weights are used only as the
  training starting point). The fine-tuned weights in `models/` were trained by this project on PlantVillage.

## Software (installed from `requirements.txt`, not bundled)
PyTorch and torchvision (BSD-3-Clause) · scikit-learn (BSD-3-Clause) · NumPy (BSD-3-Clause) · pandas (BSD-3-Clause) ·
Pillow (MIT-CMU) · Matplotlib (Matplotlib licence, PSF-based) · Streamlit (Apache-2.0) · Plotly (MIT) · PyYAML (MIT) ·
joblib (BSD-3-Clause) · pytest (MIT) · tabulate (MIT). Fonts: Fraunces and DM Sans via Google Fonts (SIL Open Font License),
loaded online with system fallbacks.

## Project code
No open-source licence has been applied to the AgriShield code yet. Add one (for example MIT) before publishing it.
