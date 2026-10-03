"""Dataset inspection, duplicate detection and leakage-safe splitting for PlantVillage."""
import hashlib
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.imaging import ALLOWED_EXTENSIONS, load_rgb


def list_images(root: Path) -> pd.DataFrame:
    """One row per image file in root/<class>/<file>."""
    rows = []
    for cls_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for f in sorted(cls_dir.iterdir()):
            if f.suffix.lower() in ALLOWED_EXTENSIONS:
                rows.append({"path": str(f.relative_to(root)), "class_name": cls_dir.name})
    return pd.DataFrame(rows)


def dhash(img: Image.Image, size: int = 8) -> int:
    """64-bit difference hash: near-identical images give hashes within a few bits."""
    g = np.asarray(img.convert("L").resize((size + 1, size), Image.BILINEAR), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def thumbnail(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("L").resize((32, 32), Image.BILINEAR), dtype=np.uint8)


def hash_images(root: Path, df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Adds md5, dhash, width, height and a read_error column (unreadable files are recorded, never skipped silently).
    Also returns 32x32 grayscale thumbnails used to confirm near-duplicates."""
    rows, thumbs = [], []
    for rel in df["path"]:
        p = root / rel
        try:
            md5 = hashlib.md5(p.read_bytes()).hexdigest()
            img = load_rgb(p)
            rows.append((md5, dhash(img), img.width, img.height, ""))
            thumbs.append(thumbnail(img))
        except Exception as e:      # recorded in the manifest and reported, never silently dropped
            rows.append(("", -1, 0, 0, str(e)))
            thumbs.append(np.zeros((32, 32), np.uint8))
    cols = list(zip(*rows)) if rows else [[]] * 5
    return (df.assign(md5=list(cols[0]), dhash=list(cols[1]), width=list(cols[2]), height=list(cols[3]),
                      read_error=list(cols[4])), np.stack(thumbs) if thumbs else np.zeros((0, 32, 32), np.uint8))


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, a):
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def source_key(path: str, class_name: str) -> str:
    """PlantVillage names files '<uuid>___<source id>.JPG'; several photos of ONE leaf share a source id with a
    '.1', '.2' ... suffix (e.g. 'GHLB2 Leaf 117.1' ... '.4'), and copies end in ' copy'. Same class + same base id
    = same physical leaf."""
    name = Path(path).name
    if "___" not in name:
        return ""
    stem = Path(name.split("___", 1)[1]).stem.strip()
    stem = re.sub(r"\s*copy$", "", stem, flags=re.IGNORECASE).strip()
    return f"{class_name}|{re.sub(r'\.\d+$', '', stem).strip()}"


def duplicate_groups(df: pd.DataFrame, thumbs: np.ndarray, max_hamming: int = 4, max_thumb_mae: float = 8.0) -> tuple[np.ndarray, dict]:
    """Group images that are exact copies (same md5), near-duplicates (dHash distance <= max_hamming) or
    photos of the same leaf (same class and source id, see source_key).

    Near-duplicate search uses the pigeonhole principle: two 64-bit hashes within 4 bits share at least one of
    5 bit-blocks exactly, so only images sharing a block are compared. A dHash match alone is too coarse for
    single leaves on plain backgrounds, so each candidate pair is CONFIRMED by a mean absolute difference <= 8 grey
    levels between 32x32 thumbnails (measured: every confirmed pair on PlantVillage was same-class)."""
    n = len(df)
    uf = UnionFind(n)
    stats = {"exact_duplicate_pairs": 0, "near_duplicate_pairs": 0, "same_leaf_links": 0,
             "near_duplicate_rule": f"dHash distance <= {max_hamming} bits AND 32x32 thumbnail MAE <= {max_thumb_mae}"}
    by_key = defaultdict(list)
    for i, (p, c) in enumerate(zip(df["path"], df["class_name"])):
        k = source_key(p, c)
        if k:
            by_key[k].append(i)
    for idx in by_key.values():
        for j in idx[1:]:
            uf.union(idx[0], j)
            stats["same_leaf_links"] += 1
    by_md5 = defaultdict(list)
    for i, m in enumerate(df["md5"]):
        if m:
            by_md5[m].append(i)
    for idx in by_md5.values():
        for j in idx[1:]:
            uf.union(idx[0], j)
            stats["exact_duplicate_pairs"] += 1

    hashes = df["dhash"].to_numpy()
    valid = hashes >= 0
    blocks = [(0, 13), (13, 26), (26, 39), (39, 52), (52, 64)]
    seen = set()
    for lo, hi in blocks:
        buckets = defaultdict(list)
        mask = (1 << (hi - lo)) - 1
        for i in np.flatnonzero(valid):
            buckets[(int(hashes[i]) >> lo) & mask].append(i)
        for idx in buckets.values():
            if len(idx) < 2:
                continue
            for a in range(len(idx)):
                for b in range(a + 1, len(idx)):
                    i, j = idx[a], idx[b]
                    if (i, j) in seen:
                        continue
                    if (bin(int(hashes[i]) ^ int(hashes[j])).count("1") <= max_hamming and
                            np.abs(thumbs[i].astype(np.float32) - thumbs[j]).mean() <= max_thumb_mae):
                        seen.add((i, j))
                        if df["md5"].iat[i] != df["md5"].iat[j]:
                            stats["near_duplicate_pairs"] += 1
                        uf.union(i, j)
    roots = np.array([uf.find(i) for i in range(n)])
    _, group_ids = np.unique(roots, return_inverse=True)
    return group_ids, stats


def assign_splits(df: pd.DataFrame, fractions: dict, seed: int) -> pd.Series:
    """Stratified (per class) split at the GROUP level: every duplicate group lands in exactly one partition.
    A group's class is its majority class; groups with mixed labels are reported separately."""
    rng = np.random.default_rng(seed)
    names = ["train", "val", "calibration", "test"]
    fr = np.array([fractions[k] for k in names])
    split = pd.Series("", index=df.index)
    group_class = df.groupby("group_id")["class_name"].agg(lambda s: s.value_counts().index[0])
    sizes = df.groupby("group_id").size()
    for cls in sorted(group_class.unique()):
        groups = group_class.index[group_class == cls].to_numpy()
        groups = groups[rng.permutation(len(groups))]
        total = sizes[groups].sum()
        targets = np.cumsum(fr) * total
        running = 0
        for g in groups:
            k = int(np.searchsorted(targets, running + sizes[g] / 2))   # place by group midpoint
            split[df["group_id"] == g] = names[min(k, 3)]
            running += sizes[g]
    return split
