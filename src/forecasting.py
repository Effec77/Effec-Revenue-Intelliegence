"""Model training, comparison, and production prediction for daily revenue."""

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

from src.features import FEATURE_COLUMNS, build_feature_frame, make_feature_row
from src.ingestion import load_transactions, prepare_daily_revenue

MODEL_VERSION = "1.0"
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
TEST_FRACTION = 0.2


def _model_filename(country: str = None) -> str:
    suffix = country.lower().replace(" ", "_") if country else "all"
    return f"revenue_model_{suffix}.joblib"


def seasonal_naive_baseline(train: pd.Series, horizon: int) -> np.ndarray:
    """Predict each future day as the revenue from 7 days earlier in the train set."""
    season = 7
    tail = train.tail(season).values
    reps = int(np.ceil(horizon / season))
    return np.tile(tail, reps)[:horizon]


def _time_split(features: pd.DataFrame, test_fraction: float = TEST_FRACTION):
    split_idx = int(len(features) * (1 - test_fraction))
    train, test = features.iloc[:split_idx], features.iloc[split_idx:]
    return train, test


def _regression_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    nonzero = y_true != 0
    mape = float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100) if nonzero.any() else None
    return {"mae": mae, "rmse": rmse, "mape": mape}


def compare_models(daily: pd.DataFrame) -> dict:
    """Train the baseline and two candidate models on a chronological split and score each."""
    features = build_feature_frame(daily)
    train, test = _time_split(features)

    X_train, y_train = train[list(FEATURE_COLUMNS)], train["revenue"]
    X_test, y_test = test[list(FEATURE_COLUMNS)], test["revenue"]

    results = {}

    baseline_preds = seasonal_naive_baseline(daily.loc[train.index, "revenue"], len(test))
    results["seasonal_naive_baseline"] = {
        "metrics": _regression_metrics(y_test, baseline_preds),
        "model": None,
    }

    linreg = LinearRegression().fit(X_train, y_train)
    results["linear_regression"] = {
        "metrics": _regression_metrics(y_test, linreg.predict(X_test)),
        "model": linreg,
    }

    gbr = GradientBoostingRegressor(random_state=42).fit(X_train, y_train)
    results["gradient_boosting"] = {
        "metrics": _regression_metrics(y_test, gbr.predict(X_test)),
        "model": gbr,
    }

    best_name = min(
        (name for name in results if results[name]["model"] is not None),
        key=lambda name: results[name]["metrics"]["rmse"],
    )

    return {"results": results, "best_model_name": best_name, "test_index": test.index}


def train_production_model(country: str = None, data_dir: str = None, models_dir: str = None) -> dict:
    """Train and persist the best-performing model for a given country (or all countries)."""
    models_dir = models_dir or MODELS_DIR
    os.makedirs(models_dir, exist_ok=True)

    transactions = load_transactions(data_dir)
    daily = prepare_daily_revenue(transactions, country)

    comparison = compare_models(daily)
    best_name = comparison["best_model_name"]
    best_model = comparison["results"][best_name]["model"]

    payload = {
        "model": best_model,
        "model_name": best_name,
        "model_version": MODEL_VERSION,
        "country": country,
        "history": daily,
        "metrics": comparison["results"][best_name]["metrics"],
    }
    joblib.dump(payload, os.path.join(models_dir, _model_filename(country)))
    return payload


def _load_or_train(country: str = None, data_dir: str = None, models_dir: str = None) -> dict:
    models_dir = models_dir or MODELS_DIR
    path = os.path.join(models_dir, _model_filename(country))
    if os.path.exists(path):
        return joblib.load(path)
    return train_production_model(country, data_dir, models_dir)


def predict_revenue(date, country: str = None, data_dir: str = None, models_dir: str = None) -> dict:
    """Predict total daily revenue for ``date``, optionally scoped to ``country``.

    ``date`` may be a string (YYYY-MM-DD) or a pandas-parseable date object.
    Raises ValueError on malformed dates or unsupported countries.
    """
    try:
        target_date = pd.to_datetime(date)
    except Exception as exc:
        raise ValueError(f"Invalid date: {date}") from exc

    if country is not None:
        transactions = load_transactions(data_dir)
        available = {c.lower() for c in transactions["country"].dropna().unique()}
        if country.lower() not in available:
            raise ValueError(f"Unsupported country: {country}")

    payload = _load_or_train(country, data_dir, models_dir)
    history = payload["history"]

    feature_row = make_feature_row(history, target_date)
    prediction = float(payload["model"].predict(feature_row[list(FEATURE_COLUMNS)])[0])
    prediction = max(prediction, 0.0)

    return {
        "date": target_date.strftime("%Y-%m-%d"),
        "country": country,
        "predicted_revenue": round(prediction, 2),
        "model_version": payload["model_version"],
        "model_name": payload["model_name"],
    }
