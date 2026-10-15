import logging
import os
from pathlib import Path

import joblib
import numpy as np
import redis
from flask import Flask, jsonify, request


MODEL_PATH = Path("/model/iris_model.joblib")
PREDICTION_COUNT_KEY = "iris:prediction_count"

artifact = joblib.load(MODEL_PATH)
model = artifact["model"]
target_names = artifact["target_names"]

redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST", "base"),
    port=int(os.getenv("REDIS_PORT", "6379")),
    decode_responses=True,
)

app = Flask(__name__)
app.logger.setLevel(logging.INFO)


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("features"), list):
        return jsonify(error="Le corps JSON doit contenir une liste 'features'."), 400

    try:
        features = np.asarray(payload["features"], dtype=float)
    except (TypeError, ValueError):
        return jsonify(error="'features' doit contenir quatre nombres."), 400

    if features.shape != (4,) or not np.isfinite(features).all():
        return jsonify(error="'features' doit contenir exactement quatre nombres finis."), 400

    prediction = int(model.predict(features.reshape(1, -1))[0])
    probabilities = model.predict_proba(features.reshape(1, -1))[0]

    try:
        prediction_count = redis_client.incr(PREDICTION_COUNT_KEY)
    except redis.RedisError:
        app.logger.exception("Impossible d'incrémenter le compteur Redis.")
        return jsonify(error="Le compteur Redis est indisponible."), 503

    return jsonify(
        prediction=prediction,
        flower=target_names[prediction],
        probabilities=probabilities.tolist(),
        prediction_count=prediction_count,
    )


@app.get("/stats")
def stats():
    try:
        prediction_count = int(redis_client.get(PREDICTION_COUNT_KEY) or 0)
    except redis.RedisError:
        app.logger.exception("Impossible de lire le compteur Redis.")
        return jsonify(error="Le compteur Redis est indisponible."), 503

    return jsonify(predictions=prediction_count)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
