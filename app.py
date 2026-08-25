"""Flask API for AAVAIL Revenue Intelligence."""

from flask import Flask, jsonify, request

from src.forecasting import predict_revenue
from src.logging_service import log_prediction, timed

app = Flask(__name__)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True) or {}
    date = payload.get("date")
    country = payload.get("country")

    if not date:
        return jsonify({"error": "'date' is required"}), 400

    with timed() as t:
        try:
            result = predict_revenue(date, country)
        except ValueError as exc:
            log_prediction(date, country, None, "unknown", "error", t.elapsed)
            return jsonify({"error": str(exc)}), 400

    log_prediction(
        result["date"], result["country"], result["predicted_revenue"],
        result["model_version"], "success", t.elapsed,
    )
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
