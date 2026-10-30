"""Train the tumor detector.

Usage:
    python scripts/train.py [--config configs/config.yaml] [--epochs N]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mri_tumor.config import load_config  # noqa: E402
from mri_tumor.train import train  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="path to a YAML config (default: configs/config.yaml)")
    parser.add_argument("--epochs", type=int, default=None, help="override train.epochs (handy for quick tests)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    train(cfg)


if __name__ == "__main__":
    main()
