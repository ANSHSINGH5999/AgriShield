"""Controlled image degradations, applied to the standardised image BEFORE model preprocessing."""
import io

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

FAMILIES = ["brightness_low", "brightness_high", "gaussian_blur", "gaussian_noise", "jpeg_compression", "low_resolution"]


def apply(img: Image.Image, family: str, value: float, seed: int = 0) -> Image.Image:
    """Return a degraded copy. `value` is the family's parameter (see config.yaml). Deterministic for a given seed."""
    if family in ("brightness_low", "brightness_high"):
        return ImageEnhance.Brightness(img).enhance(float(value))
    if family == "gaussian_blur":
        return img.filter(ImageFilter.GaussianBlur(radius=float(value)))
    if family == "gaussian_noise":
        rng = np.random.default_rng(seed)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = np.clip(arr + rng.normal(0.0, float(value), arr.shape), 0.0, 1.0)
        return Image.fromarray((arr * 255).round().astype(np.uint8))
    if family == "jpeg_compression":
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=int(value))
        buf.seek(0)
        return Image.open(buf).convert("RGB")
    if family == "low_resolution":
        w, h = img.size
        small = img.resize((max(1, round(w * value)), max(1, round(h * value))), Image.BILINEAR)
        return small.resize((w, h), Image.BILINEAR)
    raise ValueError(f"Unknown perturbation family: {family}")


def conditions(cfg: dict) -> list[tuple[str, int, float]]:
    """All evaluation conditions: ('clean', 0, 0) plus every (family, severity 1-3, value)."""
    out = [("clean", 0, 0.0)]
    for fam in FAMILIES:
        for sev, val in enumerate(cfg["perturbations"][fam], start=1):
            out.append((fam, sev, float(val)))
    return out


def apply_condition(img: Image.Image, family: str, value: float, seed: int = 0) -> Image.Image:
    return img if family == "clean" else apply(img, family, value, seed)
