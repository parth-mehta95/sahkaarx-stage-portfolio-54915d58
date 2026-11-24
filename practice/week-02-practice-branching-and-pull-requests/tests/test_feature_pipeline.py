"""
Automated unit tests for Customer Churn EDA Engine, Feature Pipeline, and Model Evaluator.
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add src to sys.path
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from eda_engine import EDAEngine
from feature_engineering import FeaturePipeline
from evaluation import ModelEvaluator


@pytest.fixture
def synthetic_churn_data() -> pd.DataFrame:
    """Provides a deterministic synthetic churn dataset for test validation."""
    np.random.seed(42)
    n = 60

    tenure = np.random.randint(1, 72, size=n)
    monthly_charges = np.random.uniform(20.0, 110.0, size=n).round(2)
    total_charges = (tenure * monthly_charges + np.random.uniform(5.0, 50.0, size=n)).round(2)

    phone_service = np.random.choice(["Yes", "No"], size=n, p=[0.9, 0.1])
    internet_service = np.random.choice(["DSL", "Fiber optic", "No"], size=n)
    online_security = np.random.choice(["Yes", "No"], size=n)
    device_protection = np.random.choice(["Yes", "No"], size=n)
    tech_support = np.random.choice(["Yes", "No"], size=n)
    streaming_tv = np.random.choice(["Yes", "No"], size=n)
    streaming_movies = np.random.choice(["Yes", "No"], size=n)
    contract = np.random.choice(["Month-to-month", "One year", "Two year"], size=n)
    payment_method = np.random.choice(
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"], size=n
    )

    # Churn logic: high monthly + month-to-month + low tenure = higher churn probability
    churn_prob = 1.0 / (1.0 + np.exp(-(monthly_charges / 50.0 - tenure / 20.0)))
    churn = np.where(np.random.rand(n) < churn_prob, "Yes", "No")

    return pd.DataFrame(
        {
            "tenure": tenure,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "PhoneService": phone_service,
            "MultipleLines": ["No"] * n,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": ["No"] * n,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaymentMethod": payment_method,
            "Churn": churn,
        }
    )


def test_eda_distribution_summary(synthetic_churn_data):
    eda = EDAEngine(target_column="Churn")
    summary = eda.compute_distribution_summary(synthetic_churn_data, ["tenure", "MonthlyCharges"])

    assert not summary.empty
    assert len(summary) == 2
    assert "mean" in summary.columns
    assert "iqr" in summary.columns
    assert summary.loc[summary["feature"] == "tenure", "count"].values[0] == 60


def test_eda_outlier_detection(synthetic_churn_data):
    eda = EDAEngine()
    outlier_res = eda.detect_outliers_iqr(synthetic_churn_data, "MonthlyCharges")

    assert "outlier_count" in outlier_res
    assert "lower_bound" in outlier_res
    assert "upper_bound" in outlier_res
    assert outlier_res["outlier_count"] >= 0


def test_eda_correlations(synthetic_churn_data):
    eda = EDAEngine(target_column="Churn")
    corr_df = eda.compute_correlations(synthetic_churn_data)

    assert not corr_df.empty
    assert "pearson_correlation" in corr_df.columns
    assert "spearman_correlation" in corr_df.columns
    assert "tenure" in corr_df["feature"].values


def test_eda_profile_generation(synthetic_churn_data):
    eda = EDAEngine(target_column="Churn")
    profile = eda.generate_eda_profile(synthetic_churn_data)

    assert profile["total_records"] == 60
    assert profile["numerical_features_count"] >= 3
    assert "class_distribution" in profile


def test_feature_pipeline_tenure_cohorts(synthetic_churn_data):
    pipeline = FeaturePipeline()
    cohorts = pipeline.create_tenure_cohorts(synthetic_churn_data)

    assert len(cohorts) == len(synthetic_churn_data)
    unique_cohorts = set(cohorts.unique())
    assert any("0-12m" in c or "12-24m" in c or "48m+" in c for c in unique_cohorts)


def test_feature_pipeline_financial_features(synthetic_churn_data):
    pipeline = FeaturePipeline()
    fin_df = pipeline.engineer_financial_features(synthetic_churn_data)

    assert "ChargeToTenureRatio" in fin_df.columns
    assert "MonthlySpendVelocity" in fin_df.columns
    assert "SpendDifferential" in fin_df.columns
    assert not fin_df["ChargeToTenureRatio"].isna().any()


def test_feature_pipeline_service_interactions(synthetic_churn_data):
    pipeline = FeaturePipeline()
    svc_df = pipeline.engineer_service_interactions(synthetic_churn_data)

    assert "TotalServicesCount" in svc_df.columns
    assert "SecuritySupportIndex" in svc_df.columns
    assert "StreamingBundleScore" in svc_df.columns
    assert (svc_df["SecuritySupportIndex"] >= 0).all()


def test_feature_pipeline_fit_and_transform(synthetic_churn_data):
    pipeline = FeaturePipeline(target_column="Churn")
    X_train, y_train = pipeline.fit_transform(synthetic_churn_data)

    assert pipeline.fitted
    assert len(X_train) == len(synthetic_churn_data)
    assert len(y_train) == len(synthetic_churn_data)
    assert not X_train.isna().any().any()
    assert y_train.isin([0, 1]).all()

    # Test transform on fresh batch
    test_batch = synthetic_churn_data.drop(columns=["Churn"]).iloc[:10]
    X_test = pipeline.transform(test_batch)
    assert X_test.shape[0] == 10
    assert X_test.shape[1] == X_train.shape[1]


def test_model_evaluator_metrics():
    y_true = [1, 0, 1, 1, 0, 0, 1, 0]
    y_pred = [1, 0, 1, 0, 0, 1, 1, 0]
    y_prob = [0.85, 0.15, 0.90, 0.45, 0.20, 0.60, 0.95, 0.10]

    metrics = ModelEvaluator.evaluate_predictions(y_true, y_pred, y_prob)

    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1_score" in metrics
    assert "roc_auc" in metrics
    assert metrics["roc_auc"] is not None


def test_risk_tiers_and_importance():
    probs = [0.95, 0.75, 0.45, 0.15, 0.05]
    tiers = ModelEvaluator.compute_risk_tiers(probs, n_tiers=5)
    assert len(tiers) == 5

    feat_names = ["MonthlySpendVelocity", "tenure", "Contract", "TotalServicesCount"]
    importances = [0.4, 0.3, 0.2, 0.1]
    rank_df = ModelEvaluator.rank_feature_importance(feat_names, importances, top_n=3)

    assert len(rank_df) == 3
    assert rank_df.iloc[0]["feature"] == "MonthlySpendVelocity"
