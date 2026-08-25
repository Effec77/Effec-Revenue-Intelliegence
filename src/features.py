"""Time-aware feature engineering for the daily revenue series."""

import numpy as np
import pandas as pd

LAGS = (1, 2, 3, 7, 14)
ROLLING_WINDOWS = (7, 14, 30)

FEATURE_COLUMNS = (
    ["dow", "day", "month", "is_weekend"]
    + [f"lag_{lag}" for lag in LAGS]
    + [f"roll_mean_{w}" for w in ROLLING_WINDOWS]
    + [f"roll_std_{w}" for w in ROLLING_WINDOWS]
    + ["orders_lag_1", "views_lag_1"]
)


def build_feature_frame(daily: pd.DataFrame) -> pd.DataFrame:
    """Expand a daily revenue/orders/views series into a supervised-learning table.

    ``daily`` must be indexed by date with ``revenue``, ``orders``, ``views`` columns
    (the shape returned by ``ingestion.prepare_daily_revenue``).
    """
    frame = daily.copy()
    frame["dow"] = frame.index.dayofweek
    frame["day"] = frame.index.day
    frame["month"] = frame.index.month
    frame["is_weekend"] = (frame["dow"] >= 5).astype(int)

    for lag in LAGS:
        frame[f"lag_{lag}"] = frame["revenue"].shift(lag)

    for window in ROLLING_WINDOWS:
        shifted = frame["revenue"].shift(1)
        frame[f"roll_mean_{window}"] = shifted.rolling(window).mean()
        frame[f"roll_std_{window}"] = shifted.rolling(window).std()

    frame["orders_lag_1"] = frame["orders"].shift(1)
    frame["views_lag_1"] = frame["views"].shift(1)

    frame = frame.dropna(subset=list(FEATURE_COLUMNS))
    return frame


def make_feature_row(history: pd.DataFrame, target_date: pd.Timestamp) -> pd.DataFrame:
    """Build a single feature row for ``target_date`` using only data strictly before it.

    Used at prediction time when the target date may be beyond the known history.
    """
    past = history[history.index < target_date]
    if past.empty:
        raise ValueError("Not enough history before the requested date to build features")

    row = {
        "dow": target_date.dayofweek,
        "day": target_date.day,
        "month": target_date.month,
        "is_weekend": int(target_date.dayofweek >= 5),
    }
    revenue_series = past["revenue"]
    for lag in LAGS:
        row[f"lag_{lag}"] = revenue_series.iloc[-lag] if len(revenue_series) >= lag else revenue_series.mean()
    for window in ROLLING_WINDOWS:
        tail = revenue_series.tail(window)
        row[f"roll_mean_{window}"] = tail.mean()
        row[f"roll_std_{window}"] = tail.std() if len(tail) > 1 else 0.0
    row["orders_lag_1"] = past["orders"].iloc[-1]
    row["views_lag_1"] = past["views"].iloc[-1]

    return pd.DataFrame([row])[list(FEATURE_COLUMNS)].fillna(0.0)
