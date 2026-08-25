import pytest

import app as app_module


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "predict_revenue", app_module.predict_revenue)
    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as client:
        yield client


@pytest.fixture(autouse=True)
def isolate_logs(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "log_prediction", lambda *a, **k: None)
    yield


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}


def test_predict_all_countries(client):
    response = client.post("/predict", json={"date": "2018-11-20"})
    assert response.status_code == 200
    body = response.get_json()
    assert body["country"] is None
    assert isinstance(body["predicted_revenue"], (int, float))


def test_predict_specific_country(client):
    response = client.post("/predict", json={"date": "2018-11-20", "country": "Australia"})
    assert response.status_code == 200
    body = response.get_json()
    assert body["country"] == "Australia"
    assert isinstance(body["predicted_revenue"], (int, float))


def test_predict_missing_date(client):
    response = client.post("/predict", json={})
    assert response.status_code == 400
    assert "error" in response.get_json()


def test_predict_invalid_date(client):
    response = client.post("/predict", json={"date": "not-a-date"})
    assert response.status_code == 400
    assert "error" in response.get_json()


def test_predict_invalid_country(client):
    response = client.post("/predict", json={"date": "2018-11-20", "country": "Narnia"})
    assert response.status_code == 400
    assert "error" in response.get_json()
