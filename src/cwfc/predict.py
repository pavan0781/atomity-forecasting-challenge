from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from cwfc.data import build_aggregate_targets, chronological_split, load_cpu_readings
from cwfc.forecasting import ModelChoice, forecast


def generate_predictions(data_dir: str | Path) -> pd.DataFrame:
    readings = load_cpu_readings(data_dir)
    targets = build_aggregate_targets(readings)
    train, holdout = chronological_split(targets, horizon=24)

    if train[
        ["active_vm_count", "aggregate_cpu_utilization"]
    ].isna().any().any():
        raise ValueError("training targets contain timestamps without valid CPU readings")

    predictions = pd.DataFrame({"timestamp": holdout["timestamp"]})
    final_models = {
        "active_vm_count": ModelChoice("persistence"),
        "aggregate_cpu_utilization": ModelChoice("autoregression", window=12),
    }
    for target, model in final_models.items():
        predictions[target] = forecast(
            train, target, len(holdout), model, holdout["timestamp"]
        ).to_numpy()

    if len(predictions) != 24:
        raise ValueError("submission must contain exactly 24 rows")
    if predictions.isna().any().any():
        raise ValueError("predictions must not contain missing values")
    if not np.isfinite(
        predictions[["active_vm_count", "aggregate_cpu_utilization"]]
    ).all().all():
        raise ValueError("predictions must be finite")
    if (predictions["active_vm_count"] < 0).any():
        raise ValueError("active_vm_count predictions must be non-negative")
    if not predictions["aggregate_cpu_utilization"].between(0, 100).all():
        raise ValueError("CPU utilization predictions must be within [0, 100]")
    predictions["active_vm_count"] = predictions["active_vm_count"].astype(int)
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate challenge holdout predictions."
    )
    parser.add_argument("--data", type=Path, default=Path("challenge_data"))
    parser.add_argument("--output", type=Path, default=Path("predictions.csv"))
    args = parser.parse_args()

    predictions = generate_predictions(args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output, index=False, lineterminator="\n")
    print(f"Wrote {len(predictions)} predictions to {args.output}")


if __name__ == "__main__":
    main()
