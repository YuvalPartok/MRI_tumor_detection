"""Evaluate the trained detector on the test patients and save prediction figures.

Usage:
    python scripts/evaluate.py [--config configs/config.yaml] [--checkpoint path] [--show]
"""

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mri_tumor.config import load_config  # noqa: E402
from mri_tumor.evaluate import evaluate  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="path to a YAML config (default: configs/config.yaml)")
    parser.add_argument("--checkpoint", default=None, help="checkpoint to evaluate (default: outputs/checkpoints/best.pt)")
    parser.add_argument("--show", action="store_true", help="open the saved figures when done (macOS)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    evaluate(cfg, Path(args.checkpoint) if args.checkpoint else None)

    if args.show:
        figures = sorted((cfg.paths.output_dir / "figures").glob("test_predictions_*.png"))
        subprocess.run(["open", *map(str, figures)], check=False)


if __name__ == "__main__":
    main()
