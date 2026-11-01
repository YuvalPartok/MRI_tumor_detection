"""Evaluate the best checkpoint on the held-out test patients and draw its predicted boxes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from mri_tumor.config import Config, get_device
from mri_tumor.data.dataset import TumorSliceDataset
from mri_tumor.data.split import load_splits
from mri_tumor.metrics import detection_metrics, paired_iou
from mri_tumor.models import build_model
from mri_tumor.train import run_inference
from mri_tumor.visualize import plot_prediction_grid


def evaluate(cfg: Config, checkpoint: Path | None = None) -> dict[str, float]:
    device = get_device(cfg.train.device)
    checkpoint = checkpoint or cfg.paths.output_dir / "checkpoints" / "best.pt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)

    model = build_model(cfg)
    model.load_state_dict(state["model"])
    model.to(device)

    test_ds = TumorSliceDataset(load_splits(cfg)["test"], cfg, augment=False)
    loader = DataLoader(test_ds, batch_size=cfg.train.batch_size, shuffle=False, num_workers=cfg.train.num_workers)
    out = run_inference(model, loader, device)
    m = detection_metrics(out["probs"], out["labels"], out["pred_boxes"], out["gt_boxes"], cfg.train.cls_threshold)

    n_pat = len(set(test_ds.patients))
    print("=" * 60)
    print(f"Test set: {n_pat} patients, {len(test_ds)} slices ({int(test_ds.labels.sum())} with tumor)")
    print(f"Checkpoint: {checkpoint} (epoch {state['epoch']})")
    print("-" * 60)
    for k, v in m.items():
        print(f"  {k:<14}: {v:.4f}")
    print("=" * 60)

    by_size = metrics_by_tumor_size(test_ds, out, cfg.train.cls_threshold)
    print("By tumor size (tumor area in the 128x128 slice):")
    print(f"  {'size':<22}{'slices':>7}{'recall':>9}{'mean IoU':>10}{'det@0.5':>9}")
    for name, s in by_size.items():
        print(f"  {name:<22}{s['n']:>7}{s['recall']:>9.3f}{s['mean_iou']:>10.3f}{s['det_rate@0.5']:>9.3f}")
    print("=" * 60)

    out_dir = cfg.paths.output_dir
    with open(out_dir / "metrics.json", "w") as f:
        json.dump({"test": m, "test_by_tumor_size": by_size, "checkpoint_epoch": state["epoch"],
                   "n_test_patients": n_pat, "n_test_slices": len(test_ds)}, f, indent=2)

    _save_figures(cfg, test_ds, out, out_dir / "figures")
    return m


# Tumor-area buckets in pixels of the 128x128 slice (1 px ~ 3.5 mm^2 of brain).
SIZE_BUCKETS = {"tiny (< 20 px)": (1, 20), "small (20-150 px)": (20, 150),
                "medium (150-600 px)": (150, 600), "large (>= 600 px)": (600, 10**9)}


def metrics_by_tumor_size(ds: TumorSliceDataset, out: dict, threshold: float) -> dict[str, dict]:
    """Recall and localization quality of tumor slices, split by how large the tumor is.

    Small tumors at the upper/lower edge of a lesion are much harder than large ones, so a single
    average hides where the model actually fails.
    """
    idx = out["indices"].numpy()
    area = ds.masks[idx].flatten(1).sum(1).numpy()
    is_tumor = out["labels"].numpy() > 0.5
    detected = out["probs"].numpy() >= threshold
    ious = np.zeros(len(idx))
    ious[is_tumor] = paired_iou(out["pred_boxes"][is_tumor], out["gt_boxes"][is_tumor]).numpy()

    result = {}
    for name, (lo, hi) in SIZE_BUCKETS.items():
        sel = is_tumor & (area >= lo) & (area < hi)
        n = int(sel.sum())
        result[name] = {
            "n": n,
            "recall": float(detected[sel].mean()) if n else 0.0,
            "mean_iou": float(ious[sel].mean()) if n else 0.0,
            "det_rate@0.5": float((detected[sel] & (ious[sel] >= 0.5)).mean()) if n else 0.0,
        }
    return result


def _save_figures(cfg: Config, ds: TumorSliceDataset, out: dict, fig_dir: Path) -> None:
    """Save grids of random test slices, half with tumor and half without."""
    rng = np.random.default_rng(cfg.data.seed)
    labels = out["labels"].numpy()
    pos, neg = np.where(labels > 0.5)[0], np.where(labels < 0.5)[0]
    per_fig = cfg.evaluate.images_per_figure
    n_pos_fig = int(round(per_fig * 0.75))  # tumor slices are the interesting ones -> 3/4 of each grid

    ious = torch.zeros(len(labels))
    ious[labels > 0.5] = paired_iou(out["pred_boxes"][labels > 0.5], out["gt_boxes"][labels > 0.5])

    for k in range(cfg.evaluate.num_figures):
        chosen = np.concatenate([rng.choice(pos, n_pos_fig, replace=False),
                                 rng.choice(neg, per_fig - n_pos_fig, replace=False)])
        rng.shuffle(chosen)
        samples = []
        for j in chosen:
            i = int(out["indices"][j])
            samples.append({
                "image": ds.images[i].float().numpy(),
                "label": float(labels[j]),
                "gt_box": out["gt_boxes"][j].numpy(),
                "prob": float(out["probs"][j]),
                "pred_box": out["pred_boxes"][j].numpy(),
                "iou": float(ious[j]),
                "patient": ds.patients[i],
                "slice": int(ds.slice_idx[i]),
            })
        path = fig_dir / f"test_predictions_{k + 1}.png"
        plot_prediction_grid(samples, path, cfg.train.cls_threshold,
                             title="Test set - green dashed: ground truth, red: model prediction")
        print(f"Saved {path}")
