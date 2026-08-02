"""Loads the trained pipeline once and exposes a simple predict function.

Kept separate from api/main.py so the prediction logic can be unit-tested
without spinning up FastAPI, and reused by a CLI or batch job later.
"""
import joblib
import pandas as pd
from functools import lru_cache
from pathlib import Path

from churn_predictor.utils.config import load_config, resolve


@lru_cache(maxsize=1)
def get_pipeline():
    cfg = load_config()
    model_path = resolve(cfg["model"]["save_path"])
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"No trained model found at {model_path}. Run training first: "
            "`uv run python -m churn_predictor.models.train`"
        )
    return joblib.load(model_path)


def predict_one(features: dict) -> dict:
    pipeline = get_pipeline()
    X = pd.DataFrame([features])
    pred = int(pipeline.predict(X)[0])
    proba = float(pipeline.predict_proba(X)[0][1])
    return {
        "churn_prediction": bool(pred),
        "churn_probability": round(proba, 4),
    }