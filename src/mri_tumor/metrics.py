"""Evaluation metrics for slice-level tumor detection + localization."""

from __future__ import annotations

import numpy as np
import torch
from torchvision.ops import box_iou


def paired_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """IoU between a[i] and b[i] for each i (boxes as x1, y1, x2, y2)."""
    if len(a) == 0:
        return torch.zeros(0)
    return box_iou(a, b).diagonal()


def detection_metrics(probs: torch.Tensor, labels: torch.Tensor, pred_boxes: torch.Tensor,
                      gt_boxes: torch.Tensor, threshold: float = 0.5) -> dict[str, float]:
    """Compute classification and localization metrics over a whole split.

    * accuracy / precision / recall / F1  - "does this slice contain a tumor?"
    * mean_iou       - average IoU between predicted and true box on tumor slices
    * det_rate@0.5   - fraction of tumor slices that are both detected (prob > threshold) and
                       localized with IoU >= 0.5 (the usual "correct detection" criterion)
    """
    pred = probs >= threshold
    gt = labels > 0.5
    tp = (pred & gt).sum().item()
    fp = (pred & ~gt).sum().item()
    fn = (~pred & gt).sum().item()
    tn = (~pred & ~gt).sum().item()

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    ious = paired_iou(pred_boxes[gt], gt_boxes[gt])
    mean_iou = ious.mean().item() if len(ious) else 0.0
    det_rate = ((ious >= 0.5) & pred[gt]).float().mean().item() if len(ious) else 0.0

    return {
        "accuracy": (tp + tn) / max(len(labels), 1),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_iou": mean_iou,
        "median_iou": float(np.median(ious.numpy())) if len(ious) else 0.0,
        "det_rate@0.5": det_rate,
    }
