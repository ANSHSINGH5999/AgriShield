"""Image loading, standardisation and model preprocessing.

Pipeline (identical for training data, test data and uploaded images):
    file -> RGB PIL image -> standardise (shorter side = 256 px) -> [optional perturbation]
         -> centre crop 224 -> tensor -> ImageNet normalisation
"""
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from torchvision import transforms

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MAX_UPLOAD_BYTES = 15 * 1024 * 1024


class ImageError(ValueError):
    """Raised for unreadable or unsupported images (message is safe to show to users)."""


def load_rgb(source) -> Image.Image:
    """Open a path or file-like object as an RGB image (EXIF rotation applied)."""
    try:
        img = Image.open(source)
        img = ImageOps.exif_transpose(img)
        return img.convert("RGB")      # handles RGBA, grayscale, palette and CMYK images
    except (UnidentifiedImageError, OSError) as e:
        raise ImageError(f"The file could not be read as an image ({e.__class__.__name__}).") from None


def standardise(img: Image.Image, size: int = 256) -> Image.Image:
    """Resize so the shorter side equals `size`, keeping the aspect ratio."""
    w, h = img.size
    scale = size / min(w, h)
    return img.resize((max(size, round(w * scale)), max(size, round(h * scale))), Image.BICUBIC)


def eval_transform(cfg: dict) -> transforms.Compose:
    im = cfg["image"]
    return transforms.Compose([
        transforms.CenterCrop(im["input_size"]),
        transforms.ToTensor(),                       # HWC uint8 RGB -> CHW float 0..1 (RGB order kept)
        transforms.Normalize(im["mean"], im["std"]),
    ])


def train_transform(cfg: dict) -> transforms.Compose:
    """Mild augmentation only: crops, flips, small rotations and slight colour jitter.
    Blur / noise / compression are NOT used, so the robustness study measures unseen degradations."""
    im = cfg["image"]
    return transforms.Compose([
        transforms.RandomResizedCrop(im["input_size"], scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(im["mean"], im["std"]),
    ])


def to_model_input(img: Image.Image, cfg: dict) -> torch.Tensor:
    """Standardised PIL image -> [1, 3, 224, 224] tensor."""
    return eval_transform(cfg)(img).unsqueeze(0)


def validate_upload(name: str, size_bytes: int) -> None:
    if Path(name).suffix.lower() not in ALLOWED_EXTENSIONS:
        raise ImageError("Please upload a JPG, JPEG or PNG image.")
    if size_bytes > MAX_UPLOAD_BYTES:
        raise ImageError(f"The image is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB. Please upload a smaller file.")
    if size_bytes == 0:
        raise ImageError("The uploaded file is empty.")


def as_array(img: Image.Image) -> np.ndarray:
    return np.asarray(img, dtype=np.float32) / 255.0
