"""Download the UTSW-Glioma data from The Cancer Imaging Archive (TCIA).

* The clinical metadata is a small TSV served over plain HTTPS.
* The imaging data (22.9 GB for all 625 patients) is only distributed through an
  IBM Aspera "Faspex" public link. We use the open-source ``ascli`` command line client
  to request *only* the files this project needs (one MRI modality + tumor masks) for
  the first ``data.num_patients`` patients, which shrinks the download to ~5 MB/patient.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from mri_tumor.config import Config

# Homebrew Ruby + its gem bin directory, where `gem install aspera-cli` puts `ascli` on macOS.
_EXTRA_PATHS = ["/opt/homebrew/opt/ruby/bin", "/opt/homebrew/lib/ruby/gems/4.0.0/bin", "/usr/local/opt/ruby/bin"]


def download_metadata(cfg: Config) -> Path:
    """Fetch the clinical metadata TSV (skipped if already present)."""
    dest = cfg.paths.metadata_file
    if dest.exists():
        print(f"[metadata] already present: {dest}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[metadata] downloading {cfg.download.metadata_url}")
    urllib.request.urlretrieve(cfg.download.metadata_url, dest)
    return dest


def select_patients(cfg: Config) -> pd.DataFrame:
    """Return the metadata rows of the first ``num_patients`` patients (sorted by ID)."""
    meta = pd.read_csv(cfg.paths.metadata_file, sep="\t")
    meta = meta.sort_values("Subject ID").reset_index(drop=True)
    return meta.head(cfg.data.num_patients)


def files_for_patient(row: pd.Series, cfg: Config) -> list[str]:
    """Names of the files we need for one patient.

    Only ~85% of patients have a manually refined mask; the others fall back to the
    automatic FeTS segmentation, so we request whichever exists.
    """
    has_manual = str(row["Manually Refined Segmentation"]).strip().lower() == "yes"
    seg = cfg.data.manual_seg if has_manual else cfg.data.fallback_seg
    return [f"{cfg.data.modality}.nii.gz", f"{seg}.nii.gz"]


def _find_ascli() -> str:
    """Locate the ``ascli`` executable, also searching Homebrew Ruby locations."""
    os.environ["PATH"] = os.pathsep.join(_EXTRA_PATHS + [os.environ.get("PATH", "")])
    exe = shutil.which("ascli")
    if exe is None:
        raise RuntimeError(
            "`ascli` (IBM Aspera CLI) not found. Install it with:\n"
            "  brew install ruby && $(brew --prefix ruby)/bin/gem install aspera-cli && ascli config ascp install\n"
            "or run scripts/download_data.sh which does this for you."
        )
    return exe


def _receive(ascli: str, url: str, sources: list[str], dest: Path) -> None:
    """Ask Faspex for a specific list of files and save them (flattened) into ``dest``."""
    ts = json.dumps({"paths": [{"source": s} for s in sources]})
    cmd = [
        ascli, "faspex5", "packages", "receive",
        f"--url={url}",
        f"--to-folder={dest}",
        "--sources=@ts",      # take the file list from the transfer spec below (not the whole package!)
        f"--ts=@json:{ts}",
        "--once-only=no",
        "--progress-bar=no",
        "--format=csv",
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def _download_patient(ascli: str, cfg: Config, row: pd.Series, max_retries: int = 3) -> str:
    """Download the missing files of one patient; returns a short status message."""
    pid = row["Subject ID"]
    dest: Path = cfg.paths.raw_dir / pid
    missing = [nm for nm in files_for_patient(row, cfg) if not (dest / nm).exists()]
    if not missing:
        return f"{pid}: already downloaded"
    dest.mkdir(parents=True, exist_ok=True)
    sources = [f"{cfg.download.package_root}/{pid}/{nm}" for nm in missing]
    for attempt in range(1, max_retries + 1):
        try:
            _receive(ascli, cfg.download.faspex_url, sources, dest)
            break
        except subprocess.CalledProcessError as err:
            if attempt == max_retries:
                raise RuntimeError(f"{pid}: download failed: {err.stderr.strip()[-300:]}") from err
            time.sleep(5 * attempt)
    size_mb = sum(f.stat().st_size for f in dest.iterdir()) / 1e6
    return f"{pid}: {', '.join(missing)} ({size_mb:.1f} MB)"


def download_images(cfg: Config) -> None:
    """Download the selected modality + mask for each selected patient.

    Patients are fetched concurrently because every Aspera session spends ~10 s on setup,
    and already-downloaded files are skipped so an interrupted run can simply be restarted.
    """
    ascli = _find_ascli()
    patients = select_patients(cfg)
    cfg.paths.raw_dir.mkdir(parents=True, exist_ok=True)

    n = len(patients)
    print(f"[images] {n} patients requested -> {cfg.paths.raw_dir}")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=cfg.download.parallel_downloads) as pool:
        futures = [pool.submit(_download_patient, ascli, cfg, row) for _, row in patients.iterrows()]
        for done, fut in enumerate(as_completed(futures), start=1):
            print(f"[images] {done:>3}/{n} {fut.result()} [{time.time() - t0:.0f}s elapsed]", flush=True)
    print(f"[images] done in {time.time() - t0:.0f}s")
