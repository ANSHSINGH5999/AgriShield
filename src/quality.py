"""Deterministic image-quality indicators computed on the standardised (256-px) image."""
import numpy as np
from PIL import Image

LAPLACIAN = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
NOISE_KERNEL = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)


def _conv3(img: np.ndarray, k: np.ndarray) -> np.ndarray:
    """'Valid' 3x3 convolution with plain numpy."""
    h, w = img.shape
    out = np.zeros((h - 2, w - 2), dtype=np.float32)
    for dy in range(3):
        for dx in range(3):
            out += k[dy, dx] * img[dy:dy + h - 2, dx:dx + w - 2]
    return out


def luma(img: Image.Image) -> np.ndarray:
    a = np.asarray(img, dtype=np.float32) / 255.0
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]


def quality_features(img: Image.Image, original_size: tuple[int, int]) -> dict[str, float]:
    """img: standardised RGB image; original_size: (width, height) of the image as uploaded."""
    y = luma(img)
    hsv = np.asarray(img.convert("HSV"), dtype=np.float32) / 255.0
    h, w = y.shape
    # Immerkaer (1996) fast noise estimate
    noise = float(np.sqrt(np.pi / 2) * np.abs(_conv3(y, NOISE_KERNEL)).sum() / (6 * (w - 2) * (h - 2)))
    return {
        "brightness": float(y.mean()),
        "contrast": float(y.std()),
        "sharpness_log_lapvar": float(np.log10(_conv3(y * 255, LAPLACIAN).var() + 1.0)),
        "noise_sigma": noise,
        "saturation": float(hsv[..., 1].mean()),
        "original_min_side": float(min(original_size)),
        "aspect_ratio": float(max(original_size) / min(original_size)),
    }
