"""Preprocess the downloaded volumes into 2D slices and create the patient-level data split.

Usage:
    python scripts/preprocess.py [--config configs/config.yaml]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mri_tumor.config import load_config  # noqa: E402
from mri_tumor.data.preprocess import preprocess_all  # noqa: E402
from mri_tumor.data.split import make_splits  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="path to a YAML config (default: configs/config.yaml)")
    args = parser.parse_args()
    cfg = load_config(args.config)

    index = preprocess_all(cfg)
    splits = make_splits(index, cfg)

    print("\nPatient-level split (no patient appears in more than one split):")
    for name, ids in splits.items():
        part = index[index.patient.isin(ids)]
        print(f"  {name:<5}: {len(ids):>3} patients | {part.n_pos.sum():>5} tumor slices | "
              f"{part.n_neg.sum():>5} tumor-free slices")


if __name__ == "__main__":
    main()
