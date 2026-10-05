"""Out-of-distribution check ("does this look like the leaf photos the model was trained on?").

Class-conditional Mahalanobis distance on EfficientNet's 1280-d penultimate features (Lee et al., 2018):
class means and one shared covariance are fitted on TRAINING features; the rejection threshold is a percentile of
VALIDATION scores. The classifier itself is unchanged."""
import json
from pathlib import Path

import numpy as np


class MahalanobisOOD:
    def fit(self, feats: np.ndarray, labels: np.ndarray, shrinkage: float = 1e-3):
        classes = np.unique(labels)
        self.means = np.stack([feats[labels == c].mean(0) for c in classes]).astype(np.float64)
        centred = feats - self.means[np.searchsorted(classes, labels)]
        cov = centred.T @ centred / len(feats)
        cov += shrinkage * np.trace(cov) / cov.shape[0] * np.eye(cov.shape[0])   # numerical stability
        self.precision = np.linalg.inv(cov)
        return self

    def score(self, feats: np.ndarray) -> np.ndarray:
        """Smallest squared Mahalanobis distance to any class mean (higher = more unusual)."""
        x = np.asarray(feats, dtype=np.float64)
        xp = x @ self.precision
        d = (xp * x).sum(1)[:, None] - 2 * xp @ self.means.T + ((self.means @ self.precision) * self.means).sum(1)[None]
        return d.min(1)

    def save(self, path: Path, threshold: float, info: dict):
        np.savez_compressed(path, means=self.means.astype(np.float32), precision=self.precision.astype(np.float32))
        path.with_suffix(".json").write_text(json.dumps({"threshold": threshold, **info}, indent=2))

    @classmethod
    def load(cls, path: Path):
        d = np.load(path)
        m = cls()
        m.means, m.precision = d["means"].astype(np.float64), d["precision"].astype(np.float64)
        m.threshold = json.loads(path.with_suffix(".json").read_text())["threshold"]
        return m
