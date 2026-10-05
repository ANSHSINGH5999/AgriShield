"""Temperature scaling (Guo et al., 2017): one number T, fitted on VALIDATION logits, makes the displayed confidence
match how often the model is actually right. It never changes which class is predicted."""
import json
from pathlib import Path

import numpy as np
import torch


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    lg, y = torch.tensor(logits, dtype=torch.float64), torch.tensor(labels)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(lg / log_t.exp(), y)
        loss.backward()
        return loss
    opt.step(closure)
    return float(log_t.detach().exp())


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def calibrate_probs(probs: np.ndarray, temperature: float) -> np.ndarray:
    """softmax(log p / T) == softmax(logits / T), so it can be applied to stored probabilities."""
    return softmax(np.log(np.clip(probs, 1e-12, 1.0)) / temperature)


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, bins: int = 15) -> float:
    conf, pred = probs.max(1), probs.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs((pred[m] == labels[m]).mean() - conf[m].mean())
    return float(ece)


def load_temperature(models_dir: Path) -> float | None:
    p = models_dir / "calibration.json"
    return json.loads(p.read_text())["temperature"] if p.exists() else None
