"""TumorDetector: a compact single-object detector for 2D MRI slices.

Architecture
------------
    1-channel slice (128x128)
      -> ResNet-18 feature extractor (ImageNet-pretrained, conv1 adapted to grayscale)
      -> 512 x 4 x 4 feature map
      -> 1x1 conv (512 -> 128) + flatten        # keeps *where* features fired, needed for localization
      -> shared FC layer (2048 -> 256)
      -> cls_head: 1 logit   "is there a tumor in this slice?"
      -> box_head: 4 values  (cx, cy, w, h) in [0, 1]

Each slice contains at most one (merged) tumor region, so we don't need anchors or NMS like a
general-purpose detector; a direct regression head is simple, fast and laptop friendly.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

# ImageNet statistics of a gray image (mean of the RGB means / stds).
_GRAY_MEAN, _GRAY_STD = 0.449, 0.226


class TumorDetector(nn.Module):
    def __init__(self, backbone: str = "resnet18", pretrained: bool = True, dropout: float = 0.2, image_size: int = 128):
        super().__init__()
        weights = "DEFAULT" if pretrained else None
        resnet = getattr(models, backbone)(weights=weights)

        # Replace the RGB input conv with a 1-channel one. Summing the pretrained RGB filters gives
        # exactly the response the original network would have to a gray image copied into 3 channels.
        old = resnet.conv1
        resnet.conv1 = nn.Conv2d(1, old.out_channels, old.kernel_size, old.stride, old.padding, bias=False)
        with torch.no_grad():
            resnet.conv1.weight.copy_(old.weight.sum(dim=1, keepdim=True))

        # Everything except the final avg-pool + classification layer.
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        feat_ch = resnet.fc.in_features
        feat_hw = image_size // 32  # ResNet downsamples by 32

        self.neck = nn.Sequential(
            nn.Conv2d(feat_ch, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128 * feat_hw * feat_hw, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.cls_head = nn.Linear(256, 1)
        self.box_head = nn.Linear(256, 4)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (B, 1, H, W) in [0, 1]. Returns (logits (B,), boxes (B, 4) as x1, y1, x2, y2 in [0, 1])."""
        x = (x - _GRAY_MEAN) / _GRAY_STD
        h = self.neck(self.features(x))
        logits = self.cls_head(h).squeeze(1)
        cx, cy, w, hgt = torch.sigmoid(self.box_head(h)).unbind(1)
        # Predicting center/size and converting guarantees x1 <= x2 and y1 <= y2.
        boxes = torch.stack([cx - w / 2, cy - hgt / 2, cx + w / 2, cy + hgt / 2], dim=1).clamp(0, 1)
        return logits, boxes


def build_model(cfg) -> TumorDetector:
    return TumorDetector(
        backbone=cfg.model.backbone,
        pretrained=cfg.model.pretrained,
        dropout=cfg.model.dropout,
        image_size=cfg.data.image_size,
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
