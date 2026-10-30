"""Training loop with verbose progress reporting, checkpointing and early stopping."""

from __future__ import annotations

import csv
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from mri_tumor.config import Config, get_device, set_seed
from mri_tumor.data.dataset import TumorSliceDataset
from mri_tumor.data.split import load_splits
from mri_tumor.losses import build_loss
from mri_tumor.metrics import detection_metrics
from mri_tumor.models import build_model, count_parameters
from mri_tumor.visualize import plot_training_curves


def model_selection_score(m: dict[str, float]) -> float:
    """Single number used to pick the best epoch: equal weight to finding and to localizing tumors."""
    return 0.5 * m["f1"] + 0.5 * m["mean_iou"]


@torch.no_grad()
def run_inference(model, loader, device, loss_fn=None) -> dict:
    """Predict on a whole loader; returns stacked outputs (on CPU) and the mean loss."""
    model.eval()
    probs, boxes, labels, gt_boxes, indices, losses = [], [], [], [], [], []
    for batch in loader:
        x = batch["image"].to(device)
        logits, pred = model(x)
        if loss_fn is not None:
            loss, _ = loss_fn(logits, pred, batch["label"].to(device), batch["box"].to(device))
            losses.append(loss.item() * len(x))
        probs.append(torch.sigmoid(logits).cpu())
        boxes.append(pred.cpu())
        labels.append(batch["label"])
        gt_boxes.append(batch["box"])
        indices.append(batch["index"])
    out = {k: torch.cat(v) for k, v in
           dict(probs=probs, pred_boxes=boxes, labels=labels, gt_boxes=gt_boxes, indices=indices).items()}
    out["loss"] = sum(losses) / len(out["labels"]) if losses else float("nan")
    return out


def train(cfg: Config) -> Path:
    """Train the detector; returns the path of the best checkpoint."""
    set_seed(cfg.data.seed)
    device = get_device(cfg.train.device)
    out_dir: Path = cfg.paths.output_dir
    ckpt_dir, log_dir, fig_dir = out_dir / "checkpoints", out_dir / "logs", out_dir / "figures"
    for d in (ckpt_dir, log_dir, fig_dir):
        d.mkdir(parents=True, exist_ok=True)

    # ---------------- data ----------------
    splits = load_splits(cfg)
    train_ds = TumorSliceDataset(splits["train"], cfg, augment=True)
    val_ds = TumorSliceDataset(splits["val"], cfg, augment=False)
    loader_kw = dict(batch_size=cfg.train.batch_size, num_workers=cfg.train.num_workers)
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **loader_kw)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kw)

    # ---------------- model ----------------
    model = build_model(cfg).to(device)
    loss_fn = build_loss(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    # Cosine decay: large steps early for fast progress, small steps late for fine box adjustment.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.train.epochs)

    print("=" * 78)
    print(f"Device            : {device}")
    print(f"Model             : {cfg.model.backbone} detector, {count_parameters(model) / 1e6:.2f}M trainable params")
    print(f"Train / val       : {len(splits['train'])} / {len(splits['val'])} patients, "
          f"{len(train_ds)} / {len(val_ds)} slices "
          f"({int(train_ds.labels.sum())} / {int(val_ds.labels.sum())} with tumor)")
    print(f"Batches per epoch : {len(train_loader)} (batch size {cfg.train.batch_size})")
    print(f"Epochs            : up to {cfg.train.epochs} (early stopping patience {cfg.train.patience})")
    print("=" * 78)

    log_path = log_dir / "train_log.csv"
    fields = ["epoch", "lr", "train_loss", "train_cls", "train_l1", "train_giou", "val_loss",
              "val_accuracy", "val_precision", "val_recall", "val_f1", "val_mean_iou", "val_det_rate@0.5",
              "score", "seconds"]
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(fields)

    best_score, best_epoch, epochs_no_improve = -1.0, 0, 0
    best_path = ckpt_dir / "best.pt"
    t_start = time.time()

    for epoch in range(1, cfg.train.epochs + 1):
        t0 = time.time()
        model.train()
        running = {"loss": 0.0, "cls": 0.0, "l1": 0.0, "giou": 0.0}
        bar = tqdm(train_loader, desc=f"Epoch {epoch:>2}/{cfg.train.epochs}", leave=False, ncols=100)
        for step, batch in enumerate(bar, start=1):
            x = batch["image"].to(device)
            logits, pred = model(x)
            loss, parts = loss_fn(logits, pred, batch["label"].to(device), batch["box"].to(device))

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            for k in running:
                running[k] += parts[k]
            bar.set_postfix(loss=f"{running['loss'] / step:.3f}")
            if step % cfg.train.log_interval == 0:
                tqdm.write(f"  epoch {epoch} | batch {step:>4}/{len(train_loader)} | "
                           f"loss {running['loss'] / step:.4f} (cls {running['cls'] / step:.3f}, "
                           f"L1 {running['l1'] / step:.4f}, GIoU {running['giou'] / step:.3f})")
        train_parts = {k: v / len(train_loader) for k, v in running.items()}
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        # ---------------- validation ----------------
        val = run_inference(model, val_loader, device, loss_fn)
        m = detection_metrics(val["probs"], val["labels"], val["pred_boxes"], val["gt_boxes"], cfg.train.cls_threshold)
        score = model_selection_score(m)
        seconds = time.time() - t0

        improved = score > best_score
        if improved:
            best_score, best_epoch, epochs_no_improve = score, epoch, 0
            torch.save({"model": model.state_dict(), "config": dict(cfg), "epoch": epoch, "val_metrics": m}, best_path)
        else:
            epochs_no_improve += 1
        torch.save({"model": model.state_dict(), "config": dict(cfg), "epoch": epoch, "val_metrics": m},
                   ckpt_dir / "last.pt")

        print(f"Epoch {epoch:>2}/{cfg.train.epochs} | {seconds:5.1f}s | lr {lr:.2e} | "
              f"train loss {train_parts['loss']:.4f} | val loss {val['loss']:.4f} | "
              f"acc {m['accuracy']:.3f} F1 {m['f1']:.3f} | mIoU {m['mean_iou']:.3f} "
              f"det@0.5 {m['det_rate@0.5']:.3f}" + ("  <- best, saved" if improved else ""))

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, lr, train_parts["loss"], train_parts["cls"], train_parts["l1"],
                                    train_parts["giou"], val["loss"], m["accuracy"], m["precision"],
                                    m["recall"], m["f1"], m["mean_iou"], m["det_rate@0.5"], score, seconds])

        if epochs_no_improve >= cfg.train.patience:
            print(f"Early stopping: no improvement for {cfg.train.patience} epochs.")
            break

    plot_training_curves(log_path, fig_dir / "training_curves.png")
    print("=" * 78)
    print(f"Training finished in {(time.time() - t_start) / 60:.1f} min. "
          f"Best epoch {best_epoch} (score {best_score:.3f}) -> {best_path}")
    print(f"Log: {log_path} | curves: {fig_dir / 'training_curves.png'}")
    return best_path
