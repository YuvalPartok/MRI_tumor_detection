"""Download the clinical metadata and the MRI volumes for the first N patients.

Usage:
    python scripts/download_data.py                    # uses configs/config.yaml
    python scripts/download_data.py --num-patients 50  # override the number of patients
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mri_tumor.config import load_config  # noqa: E402
from mri_tumor.data.download import download_images, download_metadata  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="path to a YAML config (default: configs/config.yaml)")
    parser.add_argument("--num-patients", type=int, default=None, help="override data.num_patients")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.num_patients is not None:
        cfg.data.num_patients = args.num_patients

    download_metadata(cfg)
    download_images(cfg)


if __name__ == "__main__":
    main()
