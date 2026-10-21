from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from cwfc.metrics import mae, rmse, smape


@dataclass(frozen=True)
class ModelChoice:
    name: str
    window: int | None = None
    alpha: float = 1.0


def _predict(
    values: np.ndarray, horizon: int, model: ModelChoice
) -> np.ndarray:
    if values.size == 0:
        raise ValueError("training values must not be empty")
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if not np.isfinite(values).all():
        raise ValueError("training values must be finite")

    if model.name == "persistence":
        return np.full(horizon, values[-1], dtype=float)
    if model.name == "mean":
        window = model.window
        if window is None or window <= 0 or window > values.size:
            raise ValueError("mean model window must fit within training history")
        return np.full(horizon, values[-window:].mean(), dtype=float)
    if model.name == "trend":
        window = model.window
        if window is None or window < 2 or window > values.size:
            raise ValueError("trend model window must fit within training history")
        recent = values[-window:]
        x = np.arange(window, dtype=float)
        slope, intercept = np.polyfit(x, recent, 1)
        future_x = np.arange(window, window + horizon, dtype=float)
        return intercept + slope * future_x
    if model.name == "ridge_ar":
        window = model.window
        if window is None or window < 1 or window >= values.size:
            raise ValueError("ridge autoregression window must fit within training history")
        if model.alpha <= 0:
            raise ValueError("ridge alpha must be positive")

        features = np.lib.stride_tricks.sliding_window_view(values, window + 1)
        x_train = features[:, :-1]
        y_train = features[:, -1]
        x_mean = x_train.mean(axis=0)
        x_scale = x_train.std(axis=0)
        x_scale[x_scale == 0] = 1.0
        x_centered = (x_train - x_mean) / x_scale
        y_mean = y_train.mean()
        y_centered = y_train - y_mean
        coefficients = np.linalg.solve(
            x_centered.T @ x_centered + model.alpha * np.eye(window),
            x_centered.T @ y_centered,
        )

        history = values.copy()
        forecast = np.empty(horizon, dtype=float)
        for index in range(horizon):
            x = (history[-window:] - x_mean) / x_scale
            forecast[index] = y_mean + x @ coefficients
            history = np.append(history, forecast[index])
        return forecast
    if model.name == "autoregression":
        window = model.window
        if window is None or window < 1 or window >= values.size:
            raise ValueError("autoregression window must fit within training history")
        features = np.lib.stride_tricks.sliding_window_view(values, window + 1)
        x_train = features[:, :-1]
        y_train = features[:, -1]
        design = np.column_stack((np.ones(len(x_train)), x_train))
        coefficients, _, _, _ = np.linalg.lstsq(design, y_train, rcond=None)

        history = values.copy()
        forecast = np.empty(horizon, dtype=float)
        for index in range(horizon):
            x = np.concatenate(([1.0], history[-window:]))
            forecast[index] = x @ coefficients
            history = np.append(history, forecast[index])
        return forecast
    raise ValueError(f"unknown forecasting model: {model.name}")


def candidate_models(train_size: int) -> list[ModelChoice]:
    models = [ModelChoice("persistence")]
    for window in (6, 12, 24):
        if window <= train_size:
            models.append(ModelChoice("mean", window))
            if window >= 2:
                models.append(ModelChoice("trend", window))
        if window < train_size:
            models.append(ModelChoice("ridge_ar", window))
            if window == 12:
                models.append(ModelChoice("autoregression", window))
    return models


def _forecast_values(
    train: pd.DataFrame, target: str, horizon: int, model: ModelChoice
) -> np.ndarray:
    values = pd.to_numeric(train[target], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"training target {target!r} contains missing or invalid values")
    forecast = _predict(values, horizon, model)
    if target == "active_vm_count":
        forecast = np.maximum(0, np.rint(forecast))
    elif target == "aggregate_cpu_utilization":
        forecast = np.clip(forecast, 0, 100)
    return forecast


def evaluate_candidates(
    train: pd.DataFrame,
    target: str,
    horizon: int = 24,
    initial_train_size: int = 50,
    step: int = 12,
) -> pd.DataFrame:
    """Score fixed candidates on expanding temporal folds within the training window."""
    if target not in train.columns:
        raise ValueError(f"target column not found: {target}")
    if horizon <= 0 or initial_train_size <= 0 or step <= 0:
        raise ValueError("horizon, initial_train_size, and step must be positive")
    if len(train) < initial_train_size + horizon:
        raise ValueError(
            "not enough training observations for a temporal validation fold"
        )
    if not train["timestamp"].is_monotonic_increasing:
        raise ValueError("training timestamps must be sorted chronologically")

    fold_metrics: list[dict[str, float | str | int]] = []
    origin = initial_train_size
    while origin + horizon <= len(train):
        fold_train = train.iloc[:origin]
        actual = pd.to_numeric(
            train.iloc[origin : origin + horizon][target], errors="coerce"
        ).to_numpy(dtype=float)
        if not np.isfinite(actual).all():
            raise ValueError(f"validation target {target!r} contains invalid values")

        baseline = _forecast_values(
            fold_train, target, horizon, ModelChoice("persistence")
        )
        peak_threshold = float(np.quantile(fold_train[target].to_numpy(dtype=float), 0.9))
        peak_mask = actual >= peak_threshold
        for model in candidate_models(len(fold_train)):
            prediction = _forecast_values(fold_train, target, horizon, model)
            model_mae = mae(actual, prediction)
            baseline_mae = mae(actual, baseline)
            fold_metrics.append(
                {
                    "model": model.name,
                    "window": model.window or 0,
                    "alpha": model.alpha if model.name == "ridge_ar" else 0.0,
                    "fold_origin": origin,
                    "mae": model_mae,
                    "rmse": rmse(actual, prediction),
                    "smape": smape(actual, prediction),
                    "persistence_mae": baseline_mae,
                    "peak_mae": (
                        mae(actual[peak_mask], prediction[peak_mask])
                        if peak_mask.any()
                        else np.nan
                    ),
                }
            )
        origin += step

    per_model = (
        pd.DataFrame(fold_metrics)
        .groupby(["model", "window", "alpha"], as_index=False)
        .agg(
            mae=("mae", "mean"),
            rmse=("rmse", "mean"),
            smape=("smape", "mean"),
            persistence_mae=("persistence_mae", "mean"),
            peak_mae=("peak_mae", "mean"),
            folds=("fold_origin", "nunique"),
        )
        .sort_values(["mae", "model", "window"], kind="stable")
        .reset_index(drop=True)
    )
    per_model["relative_improvement_vs_persistence"] = np.where(
        per_model["persistence_mae"] > 0,
        1 - per_model["mae"] / per_model["persistence_mae"],
        np.where(per_model["mae"] == 0, 0.0, np.nan),
    )
    return per_model


def select_model(
    train: pd.DataFrame, target: str, horizon: int = 24
) -> tuple[ModelChoice, pd.DataFrame]:
    """Select using only expanding-window folds from the public training window."""
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if len(train) < 2 * horizon:
        return ModelChoice("persistence"), pd.DataFrame(
            columns=[
                "model",
                "window",
                "alpha",
                "mae",
                "rmse",
                "smape",
                "persistence_mae",
                "peak_mae",
                "relative_improvement_vs_persistence",
                "folds",
            ]
        )
    initial_train_size = min(50, len(train) - horizon)
    results = evaluate_candidates(
        train,
        target,
        horizon=horizon,
        initial_train_size=initial_train_size,
        step=12,
    )
    best = results.iloc[0]
    window = int(best["window"])
    alpha = float(best["alpha"])
    return ModelChoice(str(best["model"]), window or None, alpha or 1.0), results


def forecast(
    train: pd.DataFrame,
    target: str,
    horizon: int,
    model: ModelChoice,
    timestamps: pd.Series,
) -> pd.Series:
    if len(timestamps) != horizon:
        raise ValueError("forecast timestamps must match the requested horizon")
    values = _forecast_values(train, target, horizon, model)
    return pd.Series(values, index=timestamps.index, dtype=float, name=target)
