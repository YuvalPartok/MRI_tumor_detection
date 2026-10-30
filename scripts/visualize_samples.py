"""Sanity check: draw ground-truth boxes on random (optionally augmented) training slices.

Use this to verify that preprocessing, box extraction and augmentation are aligned.

Usage:
    python scripts/visualize_samples.py [--augment]
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mri_tumor.config import load_config, set_seed  # noqa: E402
from mri_tumor.data.dataset import TumorSliceDataset  # noqa: E402
from mri_tumor.data.split import load_splits  # noqa: E402
from mri_tumor.visualize import plot_prediction_grid  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--augment", action="store_true", help="apply training augmentations")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.data.seed)
    ds = TumorSliceDataset(load_splits(cfg)["train"], cfg, augment=args.augment)
    samples = []
    for i in torch.randperm(len(ds))[:16].tolist():
        item = ds[i]
        samples.append({"image": item["image"][0].numpy(), "label": float(item["label"]),
                        "gt_box": item["box"].numpy(),
                        "patient": ds.patients[i], "slice": int(ds.slice_idx[i])})
    path = cfg.paths.output_dir / "figures" / f"sanity_samples{'_augmented' if args.augment else ''}.png"
    plot_prediction_grid(samples, path, title="Ground-truth boxes (sanity check)", show_predictions=False)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
