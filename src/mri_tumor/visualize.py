"""Plotting helpers: prediction grids and training curves."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render to files; works without a display (figures are shown later on demand)
import matplotlib.patches as patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Yellow vs. red stays distinguishable for red-green color blindness (deuteranopia / protanopia):
# the two also differ strongly in brightness, and ground truth is dashed while predictions are solid.
GT_COLOR = "#ffd700"    # yellow: ground truth
PRED_COLOR = "#e8000b"  # red: model prediction


def _draw_box(ax, box, size: int, color: str, style: str = "-") -> None:
    x1, y1, x2, y2 = (np.asarray(box) * size).tolist()
    ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color,
                                   linewidth=1.8, linestyle=style))


def plot_prediction_grid(samples: list[dict], path: Path, threshold: float = 0.5, title: str = "",
                         show_predictions: bool = True) -> None:
    """Save a grid of slices with ground-truth (yellow, dashed) and predicted (red, solid) boxes.

    Each sample dict needs: image (H, W), label, gt_box, patient, slice and, when
    ``show_predictions`` is set, prob and pred_box. The predicted box is only drawn when the
    model says "tumor" (prob >= threshold).
    """
    n = len(samples)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3.25 * rows))
    for ax, s in zip(np.ravel(axes), samples):
        size = s["image"].shape[-1]
        ax.imshow(s["image"], cmap="gray", vmin=0, vmax=1)
        if s["label"] > 0.5:
            _draw_box(ax, s["gt_box"], size, GT_COLOR, "--")
        truth = "tumor" if s["label"] > 0.5 else "no tumor"
        if not show_predictions:
            ax.set_title(f"{s['patient']} z={s['slice']} | {truth}", fontsize=8)
            ax.axis("off")
            continue
        predicted_tumor = s["prob"] >= threshold
        if predicted_tumor:
            _draw_box(ax, s["pred_box"], size, PRED_COLOR)

        verdict ="OK" if predicted_tumor == (s["label"] > 0.5) else "WRONG"
        iou = f" IoU={s['iou']:.2f}" if s.get("iou") is not None and s["label"] > 0.5 and predicted_tumor else ""
        ax.set_title(f"{s['patient']} z={s['slice']} | {truth}\np(tumor)={s['prob']:.2f}{iou} [{verdict}]",
                     fontsize=8, color="black" if verdict == "OK" else "#c0392b")
        ax.axis("off")
    for ax in np.ravel(axes)[n:]:
        ax.axis("off")

    handles = [patches.Patch(edgecolor=GT_COLOR, fill=False, linestyle="--", label="ground truth")]
    if show_predictions:
        handles.append(patches.Patch(edgecolor=PRED_COLOR, fill=False, label="prediction"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False)
    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_training_curves(log_csv: Path, path: Path) -> None:
    """Plot loss and validation metrics per epoch from the training log CSV."""
    log = pd.read_csv(log_csv)
    fig, (ax1, ax3, ax2) = plt.subplots(1, 3, figsize=(17, 4))
    ax1.plot(log.epoch, log.train_loss, label="train")
    ax1.plot(log.epoch, log.val_loss, label="validation")
    ax1.set(xlabel="epoch", ylabel="loss", title="Total loss")
    ax1.legend()

    # Loss components show *which* task overfits (classification vs. box regression).
    for part, color in [("cls", "tab:blue"), ("giou", "tab:green")]:
        ax3.plot(log.epoch, log[f"train_{part}"], color=color, linestyle="--", label=f"train {part}")
        ax3.plot(log.epoch, log[f"val_{part}"], color=color, label=f"val {part}")
    ax3.set(xlabel="epoch", ylabel="loss", title="Loss components (train cls includes label smoothing)")
    ax3.legend()
    for col in ["val_f1", "val_mean_iou", "val_det_rate@0.5"]:
        ax2.plot(log.epoch, log[col], label=col.replace("val_", ""))
    ax2.set(xlabel="epoch", ylabel="score", title="Validation metrics", ylim=(0, 1))
    ax2.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
