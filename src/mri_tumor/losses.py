"""Detection loss = classification loss + box regression loss (box terms only on tumor slices)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import generalized_box_iou_loss


class DetectionLoss(nn.Module):
    """Weighted sum of:

    * BCE on the "tumor present" logit,
    * SmoothL1 on box coordinates (gets every coordinate roughly right),
    * Generalized IoU loss (directly optimizes box overlap and still gives a gradient
      when the predicted and true boxes don't overlap at all).

    Tumor-free slices have no box, so the box losses are averaged over positive slices only.
    """

    def __init__(self, cls_weight: float = 1.0, l1_weight: float = 5.0, giou_weight: float = 2.0):
        super().__init__()
        self.cls_weight, self.l1_weight, self.giou_weight = cls_weight, l1_weight, giou_weight

    def forward(self, logits, pred_boxes, labels, gt_boxes) -> tuple[torch.Tensor, dict[str, float]]:
        cls_loss = F.binary_cross_entropy_with_logits(logits, labels)

        pos = labels > 0.5
        if pos.any():
            l1 = F.smooth_l1_loss(pred_boxes[pos], gt_boxes[pos], beta=0.05)
            giou = generalized_box_iou_loss(pred_boxes[pos], gt_boxes[pos], reduction="mean")
        else:
            l1 = giou = logits.new_zeros(())

        total = self.cls_weight * cls_loss + self.l1_weight * l1 + self.giou_weight * giou
        return total, {"loss": total.item(), "cls": cls_loss.item(), "l1": l1.item(), "giou": giou.item()}


def build_loss(cfg) -> DetectionLoss:
    t = cfg.train
    return DetectionLoss(t.cls_loss_weight, t.box_l1_weight, t.box_giou_weight)
