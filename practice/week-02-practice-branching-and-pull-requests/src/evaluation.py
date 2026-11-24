"""
Model and Feature Evaluation Module.
Calculates precision, recall, F1, ROC-AUC, PR-AUC, confusion matrices,
feature importance rankings, and churn decile risk tiering.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


class ModelEvaluator:
    """Evaluates classification models and derives actionable customer churn risk tiers."""

    @staticmethod
    def evaluate_predictions(
        y_true: Union[List[int], np.ndarray, pd.Series],
        y_pred: Union[List[int], np.ndarray, pd.Series],
        y_prob: Optional[Union[List[float], np.ndarray, pd.Series]] = None,
    ) -> Dict[str, Any]:
        """
        Computes comprehensive classification metrics.
        """
        y_t = np.asarray(y_true)
        y_p = np.asarray(y_pred)

        acc = float(accuracy_score(y_t, y_p))
        prec = float(precision_score(y_t, y_p, zero_division=0))
        rec = float(recall_score(y_t, y_p, zero_division=0))
        f1 = float(f1_score(y_t, y_p, zero_division=0))
        cm = confusion_matrix(y_t, y_p).tolist()

        metrics: Dict[str, Any] = {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "confusion_matrix": cm,
            "true_negatives": int(cm[0][0]),
            "false_positives": int(cm[0][1]),
            "false_negatives": int(cm[1][0]),
            "true_positives": int(cm[1][1]),
        }

        if y_prob is not None:
            y_pr = np.asarray(y_prob)
            try:
                metrics["roc_auc"] = round(float(roc_auc_score(y_t, y_pr)), 4)
            except Exception:
                metrics["roc_auc"] = None

            try:
                metrics["pr_auc"] = round(float(average_precision_score(y_t, y_pr)), 4)
            except Exception:
                metrics["pr_auc"] = None

        return metrics

    @staticmethod
    def compute_risk_tiers(
        probabilities: Union[List[float], np.ndarray, pd.Series],
        n_tiers: int = 5,
    ) -> pd.Series:
        """
        Discretizes predicted churn probabilities into operational risk tiers:
        - Tier 1: Critical Churn Risk (>80th percentile)
        - Tier 2: High Risk
        - Tier 3: Moderate Risk
        - Tier 4: Low Risk
        - Tier 5: Champion / Safe (<20th percentile)
        """
        probs = np.asarray(probabilities)
        tier_labels = [
            f"Tier_{i+1}_Risk"
            for i in reversed(range(n_tiers))
        ]  # Tier 1 = highest risk

        # Use quantiles or uniform bins
        bins = np.linspace(0.0, 1.0, n_tiers + 1)
        binned = pd.cut(
            probs,
            bins=bins,
            labels=list(reversed(tier_labels)),
            include_lowest=True,
        )
        return binned.astype(str)

    @staticmethod
    def rank_feature_importance(
        feature_names: List[str],
        importances: Union[List[float], np.ndarray],
        top_n: int = 10,
    ) -> pd.DataFrame:
        """
        Ranks top-N features by importance score.
        """
        df = pd.DataFrame(
            {"feature": feature_names, "importance": np.abs(np.asarray(importances))}
        )
        df["normalized_weight"] = (df["importance"] / df["importance"].sum()).round(4)
        df = df.sort_values(by="importance", ascending=False).reset_index(drop=True)
        return df.head(top_n)
