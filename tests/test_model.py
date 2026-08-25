import math

import pytest

from src.forecasting import predict_revenue


@pytest.fixture(scope="module")
def all_country_model(tmp_path_factory):
    return tmp_path_factory.mktemp("models_all")


def test_prediction_executes(tmp_path):
    result = predict_revenue("2018-11-20", models_dir=str(tmp_path))
    assert "predicted_revenue" in result


def test_prediction_is_numeric(tmp_path):
    result = predict_revenue("2018-11-20", models_dir=str(tmp_path))
    assert isinstance(result["predicted_revenue"], (int, float))


def test_prediction_is_finite(tmp_path):
    result = predict_revenue("2018-11-20", models_dir=str(tmp_path))
    assert math.isfinite(result["predicted_revenue"])
    assert result["predicted_revenue"] >= 0


def test_country_prediction(tmp_path):
    result = predict_revenue("2018-11-20", country="Australia", models_dir=str(tmp_path))
    assert result["country"] == "Australia"
    assert math.isfinite(result["predicted_revenue"])


def test_invalid_country_raises(tmp_path):
    with pytest.raises(ValueError):
        predict_revenue("2018-11-20", country="Narnia", models_dir=str(tmp_path))


def test_invalid_date_raises(tmp_path):
    with pytest.raises(ValueError):
        predict_revenue("not-a-date", models_dir=str(tmp_path))
