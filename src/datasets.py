"""PyTorch dataset over a manifest of image paths (used for training and evaluation)."""
from pathlib import Path

import torch
from torch.utils.data import Dataset

from src.imaging import load_rgb, standardise


class ManifestDataset(Dataset):
    def __init__(self, root: Path, paths: list[str], labels: list[int], transform, standard_size: int = 256):
        self.root, self.paths, self.labels = root, paths, labels
        self.transform, self.size = transform, standard_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = standardise(load_rgb(self.root / self.paths[i]), self.size)   # raises on a broken file (never skipped)
        return self.transform(img), torch.tensor(self.labels[i])
