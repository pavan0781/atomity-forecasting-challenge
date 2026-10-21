import numpy as np
import pandas as pd

from cwfc.forecasting import (
    ModelChoice,
    evaluate_candidates,
    forecast,
    select_model,
)
from cwfc.predict import generate_predictions


def _series_frame(size=110):
    return pd.DataFrame(
        {
            "timestamp": np.arange(size) * 300,
            "active_vm_count": 100 + np.arange(size, dtype=float) * 0.25,
            "aggregate_cpu_utilization": 35 + np.arange(size, dtype=float) * 0.05,
        }
    )


def test_forecast_respects_bounds_and_timestamp_index():
    train = _series_frame()
    future = pd.Series([33000, 33300, 33600])

    predicted_count = forecast(
        train, "active_vm_count", 3, ModelChoice("trend", 12), future
    )
    predicted_cpu = forecast(
        train, "aggregate_cpu_utilization", 3, ModelChoice("trend", 12), future
    )

    assert predicted_count.index.equals(future.index)
    assert (predicted_count >= 0).all()
    assert predicted_cpu.between(0, 100).all()


def test_ridge_autoregression_forecasts_recursively():
    train = _series_frame()
    future = pd.Series(range(24))
    result = forecast(train, "aggregate_cpu_utilization", 24, ModelChoice("ridge_ar", 12), future)

    assert len(result) == 24
    assert np.isfinite(result).all()
    assert not result.eq(train["aggregate_cpu_utilization"].iloc[-1]).all()


def test_unregularized_ar12_forecasts_recursively():
    train = _series_frame()
    result = forecast(
        train,
        "aggregate_cpu_utilization",
        24,
        ModelChoice("autoregression", 12),
        pd.Series(range(24)),
    )

    assert len(result) == 24
    assert np.isfinite(result).all()
    assert not result.eq(train["aggregate_cpu_utilization"].iloc[-1]).all()


def test_temporal_candidate_evaluation_uses_multiple_folds():
    train = _series_frame()
    results = evaluate_candidates(train, "active_vm_count")

    assert results["folds"].eq(4).all()
    assert {"mae", "rmse", "smape", "peak_mae", "alpha"}.issubset(results.columns)
    assert results["mae"].notna().all()
    assert "ridge_ar" in set(results["model"])


def test_temporal_evaluation_handles_folds_without_training_defined_peaks():
    train = pd.DataFrame(
        {
            "timestamp": np.arange(110) * 300,
            "active_vm_count": np.arange(110, 0, -1, dtype=float),
        }
    )
    results = evaluate_candidates(train, "active_vm_count")

    assert results["peak_mae"].isna().all()


def test_model_selection_falls_back_to_persistence_when_history_is_short():
    model, results = select_model(_series_frame(30), "active_vm_count")

    assert model == ModelChoice("persistence")
    assert results.empty


def test_model_selection_uses_one_fold_for_minimum_viable_history():
    model, results = select_model(_series_frame(48), "active_vm_count")

    assert model.name in {"persistence", "mean", "trend", "ridge_ar", "autoregression"}
    assert results["folds"].eq(1).all()


def test_prediction_command_supports_small_replacement_dataset(tmp_path):
    rows = []
    for timestamp in range(60):
        for vm_id in ("vm-a", "vm-b"):
            rows.append(
                {
                    "timestamp": timestamp * 300,
                    "vm_id": vm_id,
                    "avg_cpu": 25.0 + timestamp / 10,
                }
            )
    pd.DataFrame(rows).to_csv(
        tmp_path / "vm_cpu_readings.csv.gz", index=False, compression="gzip"
    )

    predictions = generate_predictions(tmp_path)

    assert predictions.columns.tolist() == [
        "timestamp",
        "active_vm_count",
        "aggregate_cpu_utilization",
    ]
    assert len(predictions) == 24
    assert predictions.timestamp.tolist() == list(range(36 * 300, 60 * 300, 300))
    assert predictions.active_vm_count.eq(2).all()
    assert predictions.aggregate_cpu_utilization.between(0, 100).all()
