# Effec Revenue Intelligence

Author: Aaditya Arya

## Overview

Effec Revenue Intelligence predicts future daily revenue for AAVAIL, a subscription-based media platform, from its historical transaction data. It covers data ingestion, exploratory analysis, feature engineering, model comparison against a baseline, a production Flask API, request logging, and automated drift monitoring.

## Business problem

AAVAIL's leadership needs a reliable, repeatable way to forecast revenue for a given future date, both company-wide and for individual countries, to support planning and resourcing decisions. This project turns raw invoice-level transaction data into a queryable revenue forecasting service.

## Data science approach

- **Ingestion** (`src/ingestion.py`): Locates every monthly invoice JSON file under `data/input/`, normalizes the inconsistent column naming used across files (e.g. `invoice`/`invoice_id`, `total_price`/`price`, `TimesViewed`/`times_viewed`), cleans and deduplicates the records, and aggregates them into a daily revenue/orders/views series — for all countries or a single one.
- **EDA** (`notebooks/exploratory_analysis.ipynb`): Revenue and order volume over time, revenue by country, monthly and day-of-week seasonality, and missing-value/descriptive statistics checks.
- **Feature engineering** (`src/features.py`): Calendar features (day, month, day-of-week, weekend flag), multiple revenue lags, rolling mean/std windows, and lagged order/view counts.
- **Baseline & model comparison** (`src/forecasting.py`, `notebooks/model_evaluation.ipynb`): A seasonal-naive baseline is compared against a Linear Regression model and a Gradient Boosting model, using a chronological (non-shuffled) train/test split and MAE/RMSE/MAPE metrics.
- **Final model**: The best-performing model on held-out RMSE is persisted with `joblib` and served through `predict_revenue(date, country=None)`.
- **API deployment** (`app.py`): Flask endpoints for health checks and revenue prediction.
- **Logging** (`src/logging_service.py`): Every prediction request is recorded as a JSON line under `logs/`.
- **Monitoring** (`src/monitoring.py`): Population Stability Index (PSI) drift detection to flag when production data distributions diverge from training data.

## Project structure

```text
AAVAIL-Revenue-Intelligence/
├── data/
│   └── input/
├── notebooks/
│   ├── exploratory_analysis.ipynb
│   └── model_evaluation.ipynb
├── src/
│   ├── __init__.py
│   ├── ingestion.py
│   ├── features.py
│   ├── forecasting.py
│   ├── logging_service.py
│   └── monitoring.py
├── tests/
│   ├── test_api.py
│   ├── test_model.py
│   └── test_logging.py
├── models/
├── logs/
├── results/
├── app.py
├── run_tests.py
├── requirements.txt
├── Dockerfile
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Run tests

```bash
python run_tests.py
```

## Run API

```bash
python app.py
```

## Prediction examples

All countries:

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"date":"2018-11-20"}'
```

Specific country:

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"date":"2018-11-20","country":"Australia"}'
```

Health check:

```bash
curl http://localhost:5000/health
```

## Docker

```bash
docker build -t aavail-revenue-intelligence .
docker run --rm -p 5000:5000 aavail-revenue-intelligence
```

## Results

Trained on the full AAVAIL transaction history with a chronological train/test split:

| Model | MAE | RMSE | MAPE |
|---|---|---|---|
| Seasonal naive (baseline) | 5406.37 | 8572.61 | 77.24% |
| Linear Regression | 4608.43 | 6311.45 | 62.75% |
| Gradient Boosting | 6198.61 | 10192.66 | 85.26% |

**Linear Regression** was selected as the production model — it beat the baseline on every metric and outperformed gradient boosting, which overfit on this relatively short daily series. See `results/model_comparison.png` and `notebooks/model_evaluation.ipynb` for the full comparison.

## Capstone Requirements

| Requirement | File(s) |
|---|---|
| Unit tests for the API | `tests/test_api.py` |
| Unit tests for the model | `tests/test_model.py` |
| Unit tests for logging | `tests/test_logging.py` |
| All tests runnable with one script | `run_tests.py` |
| Performance monitoring mechanism | `src/monitoring.py` |
| Read/write tests isolated from production resources | `tests/conftest.py`, `pytest` `tmp_path` fixtures used throughout |
| API predicts a specific country | `POST /predict` with `country` in `app.py` |
| API predicts all countries combined | `POST /predict` without `country` in `app.py` |
| Automated data ingestion | `src/ingestion.py` |
| Multiple models compared | `src/forecasting.py::compare_models`, `notebooks/model_evaluation.ipynb` |
| EDA with visualizations | `notebooks/exploratory_analysis.ipynb`, `results/*.png` |
| Working Docker configuration | `Dockerfile` |
| Visual comparison against baseline | `results/model_comparison.png` |
