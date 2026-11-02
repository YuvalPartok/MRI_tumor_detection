"""Patient-level train / validation / test split.

Slices from the same patient are highly correlated, so splitting by *slice* would leak
information and inflate test scores. We therefore split by *patient*, stratified by tumor
grade so that every split sees a similar mix of low- and high-grade gliomas.
"""

from __future__ import annotations

import json

import pandas as pd
from sklearn.model_selection import train_test_split

from mri_tumor.config import Config


def _split(ids: pd.Series, strata: pd.Series, test_size: float, seed: int):
    """train_test_split that falls back to an unstratified split if a stratum is too small."""
    try:
        return train_test_split(ids, test_size=test_size, random_state=seed, stratify=strata)
    except ValueError:
        return train_test_split(ids, test_size=test_size, random_state=seed)


def make_splits(index: pd.DataFrame, cfg: Config) -> dict[str, list[str]]:
    """Split patients into train/val/test and save the result to ``splits.json``."""
    _, val_frac, test_frac = cfg.data.split
    seed = cfg.data.seed
    strata = index["tumor_grade"].fillna(-1).astype(int)
    ids = index["patient"]

    train_val, test = _split(ids, strata, test_frac, seed)
    tv_strata = strata[ids.isin(train_val)]
    train, val = _split(train_val, tv_strata, val_frac / (1 - test_frac), seed)

    splits = {"train": sorted(train), "val": sorted(val), "test": sorted(test)}
    assert not (set(splits["train"]) & set(splits["val"]) | set(splits["train"]) & set(splits["test"])
                | set(splits["val"]) & set(splits["test"])), "patient leakage between splits!"

    with open(cfg.paths.processed_dir / "splits.json", "w") as f:
        json.dump(splits, f, indent=2)
    return splits


def load_splits(cfg: Config) -> dict[str, list[str]]:
    with open(cfg.paths.processed_dir / "splits.json") as f:
        return json.load(f)
