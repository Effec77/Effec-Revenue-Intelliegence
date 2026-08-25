"""Structured JSON-lines logging for production prediction events."""

import json
import os
import time
from datetime import datetime, timezone

DEFAULT_LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
PREDICTIONS_LOG = "predictions.log"


def _log_path(log_dir: str = None) -> str:
    log_dir = log_dir or DEFAULT_LOG_DIR
    os.makedirs(log_dir, exist_ok=True)
    return os.path.join(log_dir, PREDICTIONS_LOG)


def log_prediction(
    date: str,
    country,
    prediction,
    model_version: str,
    status: str,
    runtime_seconds: float,
    log_dir: str = None,
) -> str:
    """Append one JSON record describing a prediction request. Returns the log file path."""
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "date": date,
        "country": country,
        "prediction": prediction,
        "model_version": model_version,
        "status": status,
        "runtime_seconds": round(runtime_seconds, 4),
    }
    path = _log_path(log_dir)
    with open(path, "a") as fh:
        fh.write(json.dumps(record) + "\n")
    return path


def read_predictions(log_dir: str = None) -> list:
    path = _log_path(log_dir)
    if not os.path.exists(path):
        return []
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


class timed:
    """Context manager returning elapsed wall-clock seconds via ``.elapsed``."""

    def __enter__(self):
        self._start = time.perf_counter()
        self.elapsed = 0.0
        return self

    def __exit__(self, exc_type, exc, tb):
        self.elapsed = time.perf_counter() - self._start
        return False
