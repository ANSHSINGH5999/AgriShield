import io

import numpy as np
import pytest
import torch
from PIL import Image

from src.config import load_config
from src.imaging import ImageError, load_rgb, standardise, to_model_input, validate_upload


def test_standardise_keeps_aspect_and_short_side(leaf_image):
    s = standardise(leaf_image, 256)
    assert min(s.size) == 256 and s.size[0] > s.size[1]


def test_model_input_shape_rgb_order_and_normalisation():
    cfg = load_config()
    red = Image.new("RGB", (256, 256), (255, 0, 0))
    x = to_model_input(red, cfg)
    assert x.shape == (1, 3, 224, 224)
    mean, std = cfg["image"]["mean"], cfg["image"]["std"]
    # channel 0 must be RED (RGB order), normalised with ImageNet statistics
    assert torch.allclose(x[0, 0, 0, 0], torch.tensor((1 - mean[0]) / std[0]), atol=1e-4)
    assert torch.allclose(x[0, 1, 0, 0], torch.tensor((0 - mean[1]) / std[1]), atol=1e-4)


def test_load_rgb_converts_rgba_and_grayscale():
    for mode in ("RGBA", "L", "P"):
        buf = io.BytesIO()
        Image.new(mode, (40, 30)).save(buf, format="PNG")
        buf.seek(0)
        assert load_rgb(buf).mode == "RGB"


def test_corrupted_file_raises_friendly_error():
    with pytest.raises(ImageError):
        load_rgb(io.BytesIO(b"this is not an image"))


@pytest.mark.parametrize("name,size", [("leaf.gif", 10), ("leaf.jpg", 0), ("leaf.png", 50 * 1024 * 1024)])
def test_upload_validation_rejects_bad_files(name, size):
    with pytest.raises(ImageError):
        validate_upload(name, size)


def test_upload_validation_accepts_jpg():
    validate_upload("Leaf.JPG", 1000)
