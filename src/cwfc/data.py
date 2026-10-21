from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


READING_COLUMNS = {"timestamp", "vm_id", "avg_cpu"}


def load_cpu_readings(data_dir: str | Path) -> pd.DataFrame:
    """Load the release readings while retaining source values for auditing."""
    path = Path(data_dir) / "vm_cpu_readings.csv.gz"
    if not path.is_file():
        raise FileNotFoundError(f"CPU readings file not found: {path}")
    readings = pd.read_csv(path, compression="gzip")
    missing_columns = READING_COLUMNS.difference(readings.columns)
    if missing_columns:
        raise ValueError(f"CPU readings are missing columns: {sorted(missing_columns)}")
    return readings


def build_aggregate_targets(readings: pd.DataFrame) -> pd.DataFrame:
    """Construct active VM count and mean CPU targets without filling gaps."""
    missing_columns = READING_COLUMNS.difference(readings.columns)
    if missing_columns:
        raise ValueError(f"CPU readings are missing columns: {sorted(missing_columns)}")
    if readings.empty:
        raise ValueError("CPU readings must not be empty")

    data = readings.loc[:, ["timestamp", "vm_id", "avg_cpu"]].copy()
    data["timestamp"] = pd.to_numeric(data["timestamp"], errors="coerce")
    data["avg_cpu"] = pd.to_numeric(data["avg_cpu"], errors="coerce")
    if data["timestamp"].isna().any() or not np.isfinite(data["timestamp"]).all():
        raise ValueError("timestamps must be finite numeric values")
    if data["vm_id"].isna().any():
        raise ValueError("vm_id must not be missing")
    if data.duplicated(["timestamp", "vm_id"]).any():
        raise ValueError("duplicate (timestamp, vm_id) readings found")

    timestamps = np.sort(data["timestamp"].unique())
    if len(timestamps) > 1 and not np.all(np.diff(timestamps) == 300):
        raise ValueError("timestamps must follow the documented 300-second grid")

    valid_cpu = data["avg_cpu"].notna() & np.isfinite(data["avg_cpu"])
    valid_cpu &= data["avg_cpu"].between(0, 100, inclusive="both")
    valid = data.loc[valid_cpu]
    targets = (
        valid.groupby("timestamp", sort=True)
        .agg(
            active_vm_count=("vm_id", "nunique"),
            aggregate_cpu_utilization=("avg_cpu", "mean"),
        )
        .reindex(timestamps)
    )
    targets.index.name = "timestamp"
    targets = targets.reset_index()
    return targets


def chronological_split(
    targets: pd.DataFrame, horizon: int = 24
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reserve the final horizon timestamps; the default yields the 110/24 split."""
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if len(targets) <= horizon:
        raise ValueError("targets must contain more observations than the horizon")
    if not targets["timestamp"].is_monotonic_increasing:
        raise ValueError("targets must be sorted chronologically")
    train = targets.iloc[:-horizon].copy().reset_index(drop=True)
    holdout = targets.iloc[-horizon:].copy().reset_index(drop=True)
    return train, holdout
