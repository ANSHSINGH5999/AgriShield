import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture
def leaf_image():
    """Synthetic 300x200 RGB image with a green blob (tests code paths only, not model quality)."""
    rng = np.random.default_rng(0)
    a = np.full((200, 300, 3), 235, np.uint8)
    yy, xx = np.mgrid[:200, :300]
    mask = ((xx - 150) / 110) ** 2 + ((yy - 100) / 70) ** 2 < 1
    a[mask] = [60, 140, 70]
    a[mask & (rng.random((200, 300)) < 0.05)] = [120, 80, 40]
    return Image.fromarray(a)
