"""
Baseline modeling module for Customer Churn Prediction.
"""

from typing import Dict, Any, Tuple
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from src.config import model_config
from src.utils import setup_logger, format_metrics

logger = setup_logger("model")


class ChurnModelPipeline:
    """Baseline machine learning pipeline for customer churn classification."""

    def __init__(self, model_type: str = "random_forest"):
        self.model_type = model_type
        if model_type == "logistic_regression":
            self.model = LogisticRegression(
                max_iter=1000,
                random_state=model_config.RANDOM_STATE,
                class_weight="balanced"
            )
        else:
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=6,
                random_state=model_config.RANDOM_STATE,
                class_weight="balanced"
            )
        self.is_fitted = False

    def train(self, X_train: Any, y_train: Any) -> "ChurnModelPipeline":
        """Train the classifier."""
        logger.info(f"Training {self.model_type} on {len(X_train)} samples...")
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        logger.info("Model training completed.")
        return self

    def evaluate(self, X_test: Any, y_test: Any) -> Dict[str, float]:
        """Compute comprehensive evaluation metrics."""
        if not self.is_fitted:
            raise ValueError("Model has not been trained yet. Call train() first.")

        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, y_prob)) if len(np.unique(y_test)) > 1 else 0.5
        }
        logger.info(f"Evaluation Results: {format_metrics(metrics)}")
        return metrics
