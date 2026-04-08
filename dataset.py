"""
dataset.py – TXT-based Dual-Pixel Autofocus Dataset
=====================================================

Each line in the annotation TXT file has the format:
    SceneName LeftRawPrefix RightRawPrefix FocusIndex GTFocusIndex PatchX PatchY Temperature

This module provides:
  • ``AutofocusRecord``  – a lightweight named-tuple for one line.
  • ``AutofocusDataset`` – a ``torch.utils.data.Dataset`` that
        - parses the TXT annotation file,
        - groups records into focal stacks (same scene + patch coords),
        - loads / crops the dual-pixel RAW images,
        - applies per-patch mean-variance normalisation,
        - constructs the RL "state" tensor (image features + encodings).
"""

from __future__ import annotations

import math
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# 1.  Data record & parsing
# ---------------------------------------------------------------------------

NUM_FOCUS_POSITIONS = 70        # 0 – 69
GRID_W = 19                     # PatchX range 0-18
GRID_H = 15                     # PatchY range 0-14
DEFAULT_PATCH_SIZE = 128        # default crop size in pixels


@dataclass
class AutofocusRecord:
    """One line of the annotation TXT file."""
    scene_name: str
    left_raw_prefix: str
    right_raw_prefix: str
    focus_index: int            # 0-69
    gt_focus_index: int         # 0-69
    patch_x: int                # 0-18
    patch_y: int                # 0-14
    temperature: float


def parse_txt(txt_path: str) -> List[AutofocusRecord]:
    """Parse the annotation TXT file and return a list of records."""
    records: List[AutofocusRecord] = []
    with open(txt_path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            rec = AutofocusRecord(
                scene_name=parts[0],
                left_raw_prefix=parts[1],
                right_raw_prefix=parts[2],
                focus_index=int(parts[3]),
                gt_focus_index=int(parts[4]),
                patch_x=int(parts[5]),
                patch_y=int(parts[6]),
                temperature=float(parts[7]),
            )
            records.append(rec)
    return records


# ---------------------------------------------------------------------------
# 2.  Focal-stack grouping
# ---------------------------------------------------------------------------

def group_focal_stacks(
    records: List[AutofocusRecord],
) -> Dict[Tuple[str, int, int], List[AutofocusRecord]]:
    """Group records that share (scene, patch_x, patch_y) into focal stacks.

    Returns
    -------
    dict  :  key = (scene_name, patch_x, patch_y)
             value = list of records sorted by focus_index
    """
    groups: Dict[Tuple[str, int, int], List[AutofocusRecord]] = defaultdict(list)
    for rec in records:
        key = (rec.scene_name, rec.patch_x, rec.patch_y)
        groups[key].append(rec)
    # Sort each stack by focus_index
    for key in groups:
        groups[key].sort(key=lambda r: r.focus_index)
    return groups


# ---------------------------------------------------------------------------
# 3.  RAW loading helpers
# ---------------------------------------------------------------------------

def load_raw_image(path: str, dtype: np.dtype = np.uint16) -> np.ndarray:
    """Load a RAW image file.

    We support several common formats:
      • ``.npy`` – numpy array saved with ``np.save``.
      • ``.raw`` / ``.bin`` – flat binary; the caller may need to
        supply width/height externally (here we infer a square).
      • ``.png`` / ``.tiff`` – loaded via PIL / cv2 if available.

    Returns a 2-D or 3-D numpy array (H, W) or (H, W, C).
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        return np.load(path)
    if ext in (".png", ".tiff", ".tif", ".jpg", ".jpeg"):
        try:
            from PIL import Image
            img = Image.open(path)
            return np.array(img)
        except ImportError:
            import cv2
            return cv2.imread(path, cv2.IMREAD_UNCHANGED)
    # Default: flat binary
    data = np.fromfile(path, dtype=dtype)
    side = int(math.isqrt(data.size))
    if side * side == data.size:
        return data.reshape(side, side)
    return data


def crop_patch(
    image: np.ndarray,
    patch_x: int,
    patch_y: int,
    patch_size: int = DEFAULT_PATCH_SIZE,
    grid_w: int = GRID_W,
    grid_h: int = GRID_H,
) -> np.ndarray:
    """Crop a patch from *image* at grid coordinate (patch_x, patch_y).

    The image is divided into ``grid_w × grid_h`` cells.  The patch
    is centred on the cell and has size ``patch_size × patch_size``.
    """
    h, w = image.shape[:2]
    cell_w = w / grid_w
    cell_h = h / grid_h
    cx = int((patch_x + 0.5) * cell_w)
    cy = int((patch_y + 0.5) * cell_h)
    half = patch_size // 2
    x0 = max(cx - half, 0)
    y0 = max(cy - half, 0)
    x1 = min(x0 + patch_size, w)
    y1 = min(y0 + patch_size, h)
    # Adjust start if patch extends beyond image boundary
    x0 = max(x1 - patch_size, 0)
    y0 = max(y1 - patch_size, 0)
    return image[y0:y1, x0:x1]


# ---------------------------------------------------------------------------
# 4.  Normalisation
# ---------------------------------------------------------------------------

def normalise_patch(patch: np.ndarray) -> np.ndarray:
    r"""Per-patch normalisation:  \hat{P} = (P - \mu(P)) / \sigma(P).

    Uses standard deviation (not variance) in the denominator to keep
    the numerical range reasonable, matching the paper's Eq. (14):
        \hat{L} = (L - \mu(L)) / Var(L)
    where ``Var`` is used loosely as ``std`` in the reference code.
    """
    patch = patch.astype(np.float32)
    mu = patch.mean()
    std = patch.std() + 1e-8
    return (patch - mu) / std


# ---------------------------------------------------------------------------
# 5.  Position Encodings
# ---------------------------------------------------------------------------

def lens_position_encoding(
    focus_index: int,
    num_positions: int = NUM_FOCUS_POSITIONS,
    embed_dim: int = 16,
) -> np.ndarray:
    """Sinusoidal positional encoding for the current lens position.

    Returns a 1-D array of shape ``(embed_dim,)``.
    """
    pe = np.zeros(embed_dim, dtype=np.float32)
    pos = focus_index / max(num_positions - 1, 1)   # normalise to [0, 1]
    for i in range(embed_dim):
        if i % 2 == 0:
            pe[i] = math.sin(pos * math.pi * (2 ** (i // 2)))
        else:
            pe[i] = math.cos(pos * math.pi * (2 ** (i // 2)))
    return pe


def roi_position_encoding(
    patch_x: int,
    patch_y: int,
    grid_w: int = GRID_W,
    grid_h: int = GRID_H,
    embed_dim: int = 16,
) -> np.ndarray:
    """Sinusoidal positional encoding for the RoI grid coordinate.

    Encodes both x and y normalised to [0, 1], concatenated to give a
    vector of length ``embed_dim``.
    """
    half = embed_dim // 2
    pe = np.zeros(embed_dim, dtype=np.float32)
    nx = patch_x / max(grid_w - 1, 1)
    ny = patch_y / max(grid_h - 1, 1)
    for i in range(half):
        freq = math.pi * (2 ** i)
        if i % 2 == 0:
            pe[i] = math.sin(nx * freq)
            pe[half + i] = math.sin(ny * freq)
        else:
            pe[i] = math.cos(nx * freq)
            pe[half + i] = math.cos(ny * freq)
    return pe


# ---------------------------------------------------------------------------
# 6.  PyTorch Dataset
# ---------------------------------------------------------------------------

class AutofocusDataset(Dataset):
    """PyTorch Dataset for the dual-pixel autofocus task.

    Each sample corresponds to **one record** (one focus index of one
    patch).  The ``__getitem__`` method returns:

    state : dict
        ``"image"``   – (2, H, W) tensor  (normalised L, R patches)
        ``"lens_pe"`` – (embed_dim,) tensor
        ``"roi_pe"``  – (embed_dim,) tensor
        ``"temperature"`` – scalar tensor
    focus_index    : int   – current lens position
    gt_focus_index : int   – ground-truth best-focus position
    meta : dict
        Additional metadata (scene_name, patch_x, patch_y, etc.)
    """

    def __init__(
        self,
        txt_path: str,
        data_root: str,
        patch_size: int = DEFAULT_PATCH_SIZE,
        pe_dim: int = 16,
        raw_suffix: str = ".npy",
    ) -> None:
        super().__init__()
        self.records = parse_txt(txt_path)
        self.data_root = data_root
        self.patch_size = patch_size
        self.pe_dim = pe_dim
        self.raw_suffix = raw_suffix

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.records)

    # ------------------------------------------------------------------
    def _load_and_crop(self, prefix: str, rec: AutofocusRecord) -> np.ndarray:
        """Load a RAW file by *prefix*, crop the relevant patch."""
        path = os.path.join(self.data_root, prefix + self.raw_suffix)
        img = load_raw_image(path)
        patch = crop_patch(img, rec.patch_x, rec.patch_y, self.patch_size)
        return normalise_patch(patch)

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int):
        rec = self.records[idx]

        # --- dual-pixel patches ---
        left = self._load_and_crop(rec.left_raw_prefix, rec)
        right = self._load_and_crop(rec.right_raw_prefix, rec)

        # Ensure 2-D (H, W) for stacking
        if left.ndim == 3:
            left = left.mean(axis=-1)
        if right.ndim == 3:
            right = right.mean(axis=-1)

        image = np.stack([left, right], axis=0)  # (2, H, W)

        # --- position encodings ---
        lens_pe = lens_position_encoding(rec.focus_index, embed_dim=self.pe_dim)
        roi_pe = roi_position_encoding(rec.patch_x, rec.patch_y, embed_dim=self.pe_dim)
        temperature = np.array([rec.temperature], dtype=np.float32)

        state = {
            "image": torch.from_numpy(image),
            "lens_pe": torch.from_numpy(lens_pe),
            "roi_pe": torch.from_numpy(roi_pe),
            "temperature": torch.from_numpy(temperature),
        }
        meta = {
            "scene_name": rec.scene_name,
            "patch_x": rec.patch_x,
            "patch_y": rec.patch_y,
            "left_raw_prefix": rec.left_raw_prefix,
            "right_raw_prefix": rec.right_raw_prefix,
        }
        return state, rec.focus_index, rec.gt_focus_index, meta


# ---------------------------------------------------------------------------
# 7.  Focal-stack Dataset (for trajectory-level access)
# ---------------------------------------------------------------------------

class FocalStackDataset(Dataset):
    """A dataset that indexes by *focal stack* (scene + patch coord).

    Each sample returns the entire stack of records for a given
    (scene, patch_x, patch_y) combination, useful for building expert
    trajectories (``trajectory_builder.py``).
    """

    def __init__(
        self,
        txt_path: str,
        data_root: str,
        patch_size: int = DEFAULT_PATCH_SIZE,
        pe_dim: int = 16,
        raw_suffix: str = ".npy",
    ) -> None:
        super().__init__()
        records = parse_txt(txt_path)
        self.stacks = group_focal_stacks(records)
        self.stack_keys = list(self.stacks.keys())
        self.data_root = data_root
        self.patch_size = patch_size
        self.pe_dim = pe_dim
        self.raw_suffix = raw_suffix

    def __len__(self) -> int:
        return len(self.stack_keys)

    def get_stack_records(self, idx: int) -> List[AutofocusRecord]:
        """Return the raw records for the *idx*-th focal stack."""
        key = self.stack_keys[idx]
        return self.stacks[key]

    def __getitem__(self, idx: int):
        """Return a list of (state, focus_index, gt) for every record
        in the stack."""
        records = self.get_stack_records(idx)
        items = []
        for rec in records:
            path_l = os.path.join(self.data_root, rec.left_raw_prefix + self.raw_suffix)
            path_r = os.path.join(self.data_root, rec.right_raw_prefix + self.raw_suffix)
            if os.path.exists(path_l) and os.path.exists(path_r):
                left = normalise_patch(
                    crop_patch(load_raw_image(path_l), rec.patch_x, rec.patch_y, self.patch_size)
                )
                right = normalise_patch(
                    crop_patch(load_raw_image(path_r), rec.patch_x, rec.patch_y, self.patch_size)
                )
                if left.ndim == 3:
                    left = left.mean(axis=-1)
                if right.ndim == 3:
                    right = right.mean(axis=-1)
                image = torch.from_numpy(np.stack([left, right], axis=0))
            else:
                # Placeholder zeros when files are missing (unit-test friendly)
                image = torch.zeros(2, self.patch_size, self.patch_size)

            lens_pe = torch.from_numpy(
                lens_position_encoding(rec.focus_index, embed_dim=self.pe_dim)
            )
            roi_pe = torch.from_numpy(
                roi_position_encoding(rec.patch_x, rec.patch_y, embed_dim=self.pe_dim)
            )
            temperature = torch.tensor([rec.temperature], dtype=torch.float32)
            state = {
                "image": image,
                "lens_pe": lens_pe,
                "roi_pe": roi_pe,
                "temperature": temperature,
            }
            items.append((state, rec.focus_index, rec.gt_focus_index))
        return items
