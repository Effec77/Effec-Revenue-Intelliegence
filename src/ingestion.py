"""Automated ingestion of raw AAVAIL transaction files into a clean revenue table."""

import glob
import os

import pandas as pd

DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "input")

# Source files use inconsistent column naming across months; normalize them all
# to this canonical schema before doing anything else.
COLUMN_ALIASES = {
    "invoice": "invoice_id",
    "invoice_id": "invoice_id",
    "customer_id": "customer_id",
    "stream_id": "stream_id",
    "StreamID": "stream_id",
    "price": "price",
    "total_price": "price",
    "times_viewed": "view_count",
    "TimesViewed": "view_count",
    "country": "country",
    "year": "year",
    "month": "month",
    "day": "day",
}

REQUIRED_COLUMNS = [
    "invoice_id", "customer_id", "stream_id", "price",
    "view_count", "country", "year", "month", "day",
]


def _read_one_file(path: str) -> pd.DataFrame:
    frame = pd.read_json(path)
    frame = frame.rename(columns=COLUMN_ALIASES)
    for col in REQUIRED_COLUMNS:
        if col not in frame.columns:
            frame[col] = pd.NA
    return frame[REQUIRED_COLUMNS]


def load_transactions(data_dir: str = None) -> pd.DataFrame:
    """Locate every invoice JSON file in ``data_dir``, load and combine them."""
    data_dir = data_dir or DEFAULT_DATA_DIR
    paths = sorted(glob.glob(os.path.join(data_dir, "*.json")))
    if not paths:
        raise FileNotFoundError(f"No JSON invoice files found under {data_dir}")

    frames = [_read_one_file(path) for path in paths]
    transactions = pd.concat(frames, ignore_index=True)

    transactions["price"] = pd.to_numeric(transactions["price"], errors="coerce")
    transactions["view_count"] = pd.to_numeric(transactions["view_count"], errors="coerce").fillna(0)
    transactions["customer_id"] = pd.to_numeric(transactions["customer_id"], errors="coerce")
    transactions["country"] = transactions["country"].astype(str).str.strip()

    date_parts = transactions[["year", "month", "day"]].astype(str)
    transactions["date"] = pd.to_datetime(
        date_parts["year"] + "-" + date_parts["month"] + "-" + date_parts["day"],
        errors="coerce",
    )

    transactions = transactions.dropna(subset=["date", "price"])
    transactions = transactions[transactions["price"] > 0]
    transactions = transactions.drop_duplicates()
    transactions = transactions.drop(columns=["year", "month", "day"])

    return transactions.reset_index(drop=True)


def prepare_daily_revenue(transactions: pd.DataFrame, country: str = None) -> pd.DataFrame:
    """Aggregate transactions into a daily revenue + order-count series.

    If ``country`` is given, restrict to that country first; otherwise all
    countries are combined into a single global series.
    """
    subset = transactions
    if country is not None:
        subset = subset[subset["country"].str.lower() == country.lower()]
        if subset.empty:
            raise ValueError(f"No transactions found for country '{country}'")

    daily = (
        subset.groupby("date")
        .agg(revenue=("price", "sum"), orders=("invoice_id", "nunique"), views=("view_count", "sum"))
        .sort_index()
    )
    full_index = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(full_index).fillna(0.0)
    daily.index.name = "date"
    return daily


def list_countries(transactions: pd.DataFrame) -> list:
    return sorted(transactions["country"].dropna().unique().tolist())
