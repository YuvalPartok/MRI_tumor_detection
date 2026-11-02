"""Turn 3D NIfTI volumes into a cache of normalized 2D axial slices with tumor masks.

For every patient we:
  1. load the skull-stripped MRI volume and its tumor segmentation (240 x 240 x 155, 1 mm, LPS),
  2. normalize intensities using brain voxels only (MRI intensities have no absolute scale),
  3. cut the volume into axial slices and drop slices that contain almost no brain,
  4. label each slice "tumor" / "no tumor" (optionally subsampling tumor-free slices),
  5. resize to ``image_size`` and save images + masks to one compressed ``.npz`` per patient.

Bounding boxes are *not* stored here; they are derived from the masks on the fly in the
Dataset, which keeps them exactly aligned with the image after data augmentation.
"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from mri_tumor.config import Config
from mri_tumor.data.download import files_for_patient, select_patients


def load_patient(raw_dir: Path, row: pd.Series, cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    """Load (image, binary tumor mask) for a patient, both shaped (H, W, D)."""
    image_name, seg_name = files_for_patient(row, cfg)
    image = np.asarray(nib.load(raw_dir / image_name).dataobj, dtype=np.float32)
    seg = np.asarray(nib.load(raw_dir / seg_name).dataobj)
    # Labels 1 (necrotic core), 2 (edema/infiltration) and 4 (enhancing tumor) are merged:
    # the detector localizes the whole tumor region.
    return image, seg > 0


def normalize_volume(volume: np.ndarray) -> np.ndarray:
    """Scale intensities to [0, 1] using robust percentiles of brain voxels.

    Background is exactly 0 in skull-stripped scans, so it is excluded from the statistics.
    Clipping at the 0.5th/99.5th percentile removes outliers (e.g. vessels, artifacts) that would
    otherwise squash the useful intensity range.
    """
    brain = volume[volume > 0]
    lo, hi = np.percentile(brain, [0.5, 99.5])
    out = np.clip((volume - lo) / (hi - lo + 1e-8), 0.0, 1.0)
    out[volume <= 0] = 0.0
    return out.astype(np.float32)


def resize_slices(slices: np.ndarray, size: int, is_mask: bool) -> np.ndarray:
    """Resize a stack of 2D slices (N, H, W) to (N, size, size)."""
    t = torch.from_numpy(slices.astype(np.float32)).unsqueeze(1)
    if is_mask:
        # "area" averaging + threshold keeps small tumors that nearest-neighbour could drop.
        area = F.interpolate(t, size=(size, size), mode="area").squeeze(1)
        out = area > 0.25
        # Tiny tumors (a few pixels) may vanish at the 0.25 threshold: keep any covered pixel instead.
        vanished = (out.flatten(1).sum(1) == 0) & (area.flatten(1).sum(1) > 0)
        out[vanished] = area[vanished] > 0
        return out.numpy()
    return F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False, antialias=True).squeeze(1).numpy()


def process_patient(row: pd.Series, cfg: Config, rng: np.random.Generator) -> dict | None:
    """Create the slice cache for a single patient. Returns summary stats (or None if skipped)."""
    pid = row["Subject ID"]
    raw_dir = cfg.paths.raw_dir / pid
    try:
        volume, mask = load_patient(raw_dir, row, cfg)
    except FileNotFoundError:
        tqdm.write(f"[preprocess] {pid}: files missing, skipped")
        return None

    volume = normalize_volume(volume)
    # Axial slices are volume[:, :, z]. Transposing gives (rows = anterior->posterior,
    # cols = patient right->left), i.e. the standard radiological display orientation.
    volume = volume.transpose(2, 1, 0)  # (D, H, W)
    mask = mask.transpose(2, 1, 0)

    hw = volume.shape[1] * volume.shape[2]
    brain_frac = (volume > 0).reshape(len(volume), -1).sum(1) / hw
    tumor_px = mask.reshape(len(mask), -1).sum(1)

    valid = brain_frac >= cfg.data.min_brain_fraction
    pos_idx = np.where(valid & (tumor_px >= cfg.data.min_tumor_pixels))[0]
    # Slices with a few tumor pixels (0 < n < min_tumor_pixels) are ambiguous and left out entirely.
    neg_idx = np.where(valid & (tumor_px == 0))[0]

    # Optionally subsample tumor-free slices to a fixed ratio; null in the config keeps them all.
    if cfg.data.neg_pos_ratio is not None:
        n_neg = min(len(neg_idx), int(round(cfg.data.neg_pos_ratio * max(len(pos_idx), 1))))
        neg_idx = np.sort(rng.choice(neg_idx, size=n_neg, replace=False)) if n_neg > 0 else neg_idx[:0]

    keep = np.concatenate([pos_idx, neg_idx])
    labels = np.concatenate([np.ones(len(pos_idx)), np.zeros(len(neg_idx))]).astype(np.uint8)

    images = resize_slices(volume[keep], cfg.data.image_size, is_mask=False)
    masks = resize_slices(mask[keep], cfg.data.image_size, is_mask=True)
    # Negative slices must have empty masks; positive ones must keep at least one pixel after resizing.
    masks[labels == 0] = False

    np.savez_compressed(
        cfg.paths.processed_dir / "slices" / f"{pid}.npz",
        images=images.astype(np.float16),  # float16 halves disk/RAM use; precision is ample for [0,1] images
        masks=masks.astype(bool),
        labels=labels,
        slice_idx=keep.astype(np.int16),
    )
    return {"patient": pid, "n_pos": int(len(pos_idx)), "n_neg": int(len(neg_idx))}


def preprocess_all(cfg: Config) -> pd.DataFrame:
    """Preprocess every selected patient and write ``index.csv`` with per-patient slice counts."""
    out_dir = cfg.paths.processed_dir / "slices"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.data.seed)

    patients = select_patients(cfg)
    rows = []
    for _, row in tqdm(patients.iterrows(), total=len(patients), desc="Preprocessing patients"):
        stats = process_patient(row, cfg, rng)
        if stats is not None:
            stats["tumor_grade"] = row["Tumor Grade"]
            rows.append(stats)

    index = pd.DataFrame(rows)
    index.to_csv(cfg.paths.processed_dir / "index.csv", index=False)
    print(f"[preprocess] {len(index)} patients, {index.n_pos.sum()} tumor slices, "
          f"{index.n_neg.sum()} tumor-free slices -> {out_dir}")
    return index
