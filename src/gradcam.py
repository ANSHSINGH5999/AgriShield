"""Grad-CAM (Selvaraju et al., 2017): which parts of the image pushed EfficientNet towards its predicted class.
Uses the last convolutional block; read-only (no weights change)."""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch
from matplotlib import colormaps
from PIL import Image


def gradcam(model, x: torch.Tensor, class_index: int, device: str) -> np.ndarray:
    """x: [1, 3, 224, 224] normalised input. Returns a 224x224 heat map scaled to 0..1."""
    store = {}

    def hook(_module, _inp, out):
        out.retain_grad()
        store["act"] = out

    handle = model.features[-1].register_forward_hook(hook)
    try:
        with torch.enable_grad():
            logits = model(x.to(device))
            model.zero_grad(set_to_none=True)
            logits[0, class_index].backward()
        act, grad = store["act"], store["act"].grad
        cam = torch.relu((grad.mean(dim=(2, 3), keepdim=True) * act).sum(1, keepdim=True))
        cam = torch.nn.functional.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam.detach().float().cpu().numpy()
    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)
    return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)


def overlay(image: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """Blend a heat map (red = most important) over the 224x224 image the model saw."""
    heat = (colormaps["jet"](cam)[..., :3] * 255).astype(np.float32)
    base = np.asarray(image.convert("RGB").resize(cam.shape[::-1]), dtype=np.float32)
    return Image.fromarray(((1 - alpha) * base + alpha * heat).clip(0, 255).astype(np.uint8))
