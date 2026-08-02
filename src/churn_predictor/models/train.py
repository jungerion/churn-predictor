"""End-to-end training entrypoint.

Run with:  uv run python -m churn_predictor.models.train
"""
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from churn_predictor.data.load import load_and_clean, train_test_columns
from churn_predictor.features.build_features import build_preprocessor, build_pipeline
from churn_predictor.models.evaluate import compute_metrics
from churn_predictor.utils.config import load_config, resolve
from churn_predictor.utils.logging import get_logger

logger = get_logger(__name__)


def get_model(cfg: dict):
    model_type = cfg["model"]["type"]
    if model_type == "random_forest":
        params = cfg["model"]["random_forest"]
        return RandomForestClassifier(**params)
    elif model_type == "logistic_regression":
        params = cfg["model"]["logistic_regression"]
        return LogisticRegression(**params)
    raise ValueError(f"Unknown model type: {model_type}")


def main():
    cfg = load_config()

    logger.info("Loading and cleaning data...")
    raw_path = resolve(cfg["data"]["raw_path"])
    df = load_and_clean(raw_path, target_column=cfg["data"]["target_column"])

    X, y = train_test_columns(
        df, cfg["data"]["target_column"], cfg["data"]["id_column"]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg["split"]["test_size"],
        random_state=cfg["split"]["random_state"],
        stratify=y,
    )

    logger.info("Building pipeline (preprocessing + model)...")
    preprocessor = build_preprocessor(X_train)
    model = get_model(cfg)
    pipeline = build_pipeline(preprocessor, model)

    logger.info("Training...")
    pipeline.fit(X_train, y_train)

    logger.info("Evaluating on held-out test set...")
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, y_pred, y_proba)
    logger.info(f"Test metrics: {metrics}")

    save_path = resolve(cfg["model"]["save_path"])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, save_path)
    logger.info(f"Saved trained pipeline to {save_path}")

    return metrics


if __name__ == "__main__":
    main()