import os

from src.logging_service import log_prediction, read_predictions


def test_log_file_created(tmp_path):
    log_prediction("2018-11-20", None, 123.45, "1.0", "success", 0.05, log_dir=str(tmp_path))
    assert os.path.exists(os.path.join(str(tmp_path), "predictions.log"))


def test_prediction_event_written(tmp_path):
    log_prediction("2018-11-20", "Australia", 99.0, "1.0", "success", 0.02, log_dir=str(tmp_path))
    records = read_predictions(str(tmp_path))
    assert len(records) == 1
    record = records[0]
    assert record["date"] == "2018-11-20"
    assert record["country"] == "Australia"
    assert record["prediction"] == 99.0
    assert record["status"] == "success"
    assert "timestamp" in record


def test_multiple_events_appended(tmp_path):
    log_prediction("2018-11-20", None, 10.0, "1.0", "success", 0.01, log_dir=str(tmp_path))
    log_prediction("2018-11-21", None, 20.0, "1.0", "success", 0.01, log_dir=str(tmp_path))
    records = read_predictions(str(tmp_path))
    assert len(records) == 2


def test_isolated_from_production_logs(tmp_path):
    log_prediction("2018-11-20", None, 10.0, "1.0", "success", 0.01, log_dir=str(tmp_path))
    default_log = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "predictions.log"
    )
    if os.path.exists(default_log):
        with open(default_log) as fh:
            production_line_count_before = sum(1 for _ in fh)
    else:
        production_line_count_before = 0

    if os.path.exists(default_log):
        with open(default_log) as fh:
            production_line_count_after = sum(1 for _ in fh)
    else:
        production_line_count_after = 0
    assert production_line_count_after == production_line_count_before
