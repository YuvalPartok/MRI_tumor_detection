"""PyTorch Dataset that serves (image, label, box) triplets from the preprocessed slice cache.

Boxes are computed from the tumor mask *after* augmentation, so rotations/zooms/flips can
never misalign the target box with the image.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from mri_tumor.config import Config


def mask_to_box(mask: torch.Tensor) -> torch.Tensor:
    """Tight bounding box (x1, y1, x2, y2) of a binary (H, W) mask, normalized to [0, 1].

    Returns zeros for an empty mask (tumor-free slice).
    """
    h, w = mask.shape
    ys, xs = torch.where(mask)
    if len(xs) == 0:
        return torch.zeros(4)
    # +1 on the max edge: a box around pixel column c spans [c, c+1).
    return torch.tensor([xs.min() / w, ys.min() / h, (xs.max() + 1) / w, (ys.max() + 1) / h], dtype=torch.float32)


class SliceAugmenter:
    """Random geometric + intensity augmentation applied identically to an image and its mask."""

    def __init__(self, cfg: Config):
        self.a = cfg.augment

    def __call__(self, image: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        a = self.a
        if torch.rand(1).item() < a.hflip_prob:
            image, mask = image.flip(-1), mask.flip(-1)

        # One random affine (rotation + zoom + shift) used for both tensors.
        angle = math.radians((torch.rand(1).item() * 2 - 1) * a.max_rotation_deg)
        scale = 1 + (torch.rand(1).item() * 2 - 1) * a.max_scale
        tx, ty = ((torch.rand(2) * 2 - 1) * a.max_translate * 2).tolist()  # grid coords span [-1, 1] -> x2
        cos, sin = math.cos(angle) / scale, math.sin(angle) / scale
        theta = torch.tensor([[cos, -sin, tx], [sin, cos, ty]], dtype=torch.float32).unsqueeze(0)
        grid = F.affine_grid(theta, size=(1, 1, *image.shape[-2:]), align_corners=False)
        image = F.grid_sample(image[None], grid, mode="bilinear", padding_mode="zeros", align_corners=False)[0]
        mask = F.grid_sample(mask[None].float(), grid, mode="nearest", padding_mode="zeros", align_corners=False)[0] > 0.5

        # Brightness / contrast jitter mimics intensity differences between scanners.
        # Applied to brain pixels only, so the background stays exactly 0 as in real skull-stripped scans.
        contrast = 1 + (torch.rand(1).item() * 2 - 1) * a.intensity_jitter
        brightness = (torch.rand(1).item() * 2 - 1) * a.intensity_jitter
        brain = image > 0
        image = torch.where(brain, (image * contrast + brightness).clamp(0, 1), image)

        # Gamma correction changes contrast non-linearly (dark vs. bright tissue), as different
        # scanners and protocols do.
        gamma = 1 + (torch.rand(1).item() * 2 - 1) * a.max_gamma
        image = torch.where(brain, image.clamp_min(1e-6) ** gamma, image)

        # A slight blur imitates thicker / lower-resolution acquisitions.
        if torch.rand(1).item() < a.blur_prob:
            image = _gaussian_blur(image, sigma=0.5 + torch.rand(1).item())

        # Additive Gaussian noise inside the brain imitates scanner noise.
        noise = torch.randn_like(image) * a.noise_std
        image = torch.where(brain, (image + noise).clamp(0, 1), image)
        return image, mask


def _gaussian_blur(image: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur of a (1, H, W) image."""
    radius = max(1, int(round(2 * sigma)))
    x = torch.arange(-radius, radius + 1, dtype=torch.float32)
    k = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    k = (k / k.sum()).view(1, 1, 1, -1)
    out = F.conv2d(image[None], k, padding=(0, radius))
    out = F.conv2d(out, k.transpose(-1, -2), padding=(radius, 0))
    return out[0]


class TumorSliceDataset(Dataset):
    """All cached slices of a list of patients, kept in memory (a few hundred MB for 150 patients)."""

    def __init__(self, patient_ids: list[str], cfg: Config, augment: bool = False):
        images, masks, labels, patients, slice_idx = [], [], [], [], []
        for pid in patient_ids:
            d = np.load(cfg.paths.processed_dir / "slices" / f"{pid}.npz")
            images.append(d["images"])
            masks.append(d["masks"])
            labels.append(d["labels"])
            slice_idx.append(d["slice_idx"])
            patients += [pid] * len(d["labels"])
        self.images = torch.from_numpy(np.concatenate(images))
        self.masks = torch.from_numpy(np.concatenate(masks))
        self.labels = torch.from_numpy(np.concatenate(labels)).float()
        self.slice_idx = np.concatenate(slice_idx)
        self.patients = patients
        self.augmenter = SliceAugmenter(cfg) if augment else None

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        image = self.images[i].float().unsqueeze(0)  # (1, H, W)
        mask = self.masks[i].unsqueeze(0)
        if self.augmenter is not None:
            image, mask = self.augmenter(image, mask)
        mask = mask[0]
        # A strong shift may push a small tumor out of view; the label follows what is visible.
        label = self.labels[i] if (self.labels[i] == 0 or mask.any()) else torch.tensor(0.0)
        return {"image": image, "label": label, "box": mask_to_box(mask), "index": i}
