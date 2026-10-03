"""EfficientNet-B0 classifier: build, save, load (safely) and predict."""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

CHECKPOINT = "efficientnet_b0.pt"
CLASS_NAMES = "class_names.json"


class ModelFileError(FileNotFoundError):
    """A required model file is missing (message is safe to show to users)."""


def build_model(num_classes: int, pretrained: bool) -> nn.Module:
    """Torchvision EfficientNet-B0; ImageNet weights only when training starts (never needed at inference)."""
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


def save_classifier(model: nn.Module, class_names: list[str], models_dir: Path) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, models_dir / CHECKPOINT)
    (models_dir / CLASS_NAMES).write_text(json.dumps(class_names, indent=2))


def load_classifier(models_dir: Path, device: str = "cpu") -> tuple[nn.Module, list[str]]:
    for name in (CHECKPOINT, CLASS_NAMES):
        if not (models_dir / name).exists():
            raise ModelFileError(f"Missing model file: models/{name}. Train the model (train_windows.bat) "
                                 f"or copy the trained 'models' folder into AgriShield.")
    class_names = json.loads((models_dir / CLASS_NAMES).read_text())
    model = build_model(len(class_names), pretrained=False)
    state = torch.load(models_dir / CHECKPOINT, map_location="cpu", weights_only=True)   # tensors only, no pickled code
    model.load_state_dict(state)
    return model.to(device).eval(), class_names


@torch.no_grad()
def predict_probs(model: nn.Module, batch: torch.Tensor, device: str) -> np.ndarray:
    return torch.softmax(model(batch.to(device)), dim=1).float().cpu().numpy()
