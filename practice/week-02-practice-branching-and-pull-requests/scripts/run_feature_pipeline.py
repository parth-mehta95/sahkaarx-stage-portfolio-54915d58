"""
Standalone runner script executing EDA, Feature Pipeline, Model Training,
and Performance Evaluation for the Customer Churn Intelligence Platform.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from eda_engine import EDAEngine
from feature_engineering import FeaturePipeline
from evaluation import ModelEvaluator


def generate_synthetic_dataset(n_samples: int = 500) -> pd.DataFrame:
    """Generates synthetic customer churn dataset for pipeline verification."""
    np.random.seed(101)
    tenure = np.random.randint(1, 72, size=n_samples)
    monthly_charges = np.random.uniform(18.0, 118.0, size=n_samples).round(2)
    total_charges = (tenure * monthly_charges + np.random.uniform(0.0, 30.0, size=n_samples)).round(2)

    phone_service = np.random.choice(["Yes", "No"], size=n_samples, p=[0.9, 0.1])
    internet_service = np.random.choice(["DSL", "Fiber optic", "No"], size=n_samples, p=[0.4, 0.45, 0.15])
    online_security = np.random.choice(["Yes", "No"], size=n_samples, p=[0.35, 0.65])
    device_protection = np.random.choice(["Yes", "No"], size=n_samples, p=[0.38, 0.62])
    tech_support = np.random.choice(["Yes", "No"], size=n_samples, p=[0.32, 0.68])
    streaming_tv = np.random.choice(["Yes", "No"], size=n_samples, p=[0.42, 0.58])
    streaming_movies = np.random.choice(["Yes", "No"], size=n_samples, p=[0.44, 0.56])
    contract = np.random.choice(["Month-to-month", "One year", "Two year"], size=n_samples, p=[0.55, 0.25, 0.20])
    payment_method = np.random.choice(
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"], size=n_samples
    )

    # Churn probability based on customer traits
    score = (
        (monthly_charges / 60.0)
        - (tenure / 25.0)
        + (1.2 * (contract == "Month-to-month"))
        - (0.8 * (online_security == "Yes"))
        - (0.7 * (tech_support == "Yes"))
    )
    prob = 1.0 / (1.0 + np.exp(-score))
    churn = np.where(np.random.rand(n_samples) < prob, "Yes", "No")

    return pd.DataFrame(
        {
            "tenure": tenure,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "PhoneService": phone_service,
            "MultipleLines": ["No"] * n_samples,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": ["No"] * n_samples,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaymentMethod": payment_method,
            "Churn": churn,
        }
    )


def main():
    print("=" * 80)
    print("Customer Churn Intelligence Platform: EDA & Feature Engineering Pipeline")
    print("=" * 80)

    # 1. Dataset Generation
    print("\n[Step 1/5] Ingesting & Generating Dataset...")
    df = generate_synthetic_dataset(n_samples=500)
    print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Churn Distribution: {dict(df['Churn'].value_counts())}")

    # 2. EDA Profiling
    print("\n[Step 2/5] Running Automated EDA Engine...")
    eda = EDAEngine(target_column="Churn")
    profile = eda.generate_eda_profile(df)
    print(f"Numerical Features: {profile['numerical_features']}")
    print("Top Feature Correlations with Churn:")
    for corr in profile["top_correlations"]:
        print(f"  - {corr['feature']}: Pearson={corr['pearson_correlation']}, Spearman={corr['spearman_correlation']}")

    # 3. Feature Pipeline Transformation
    print("\n[Step 3/5] Executing Feature Engineering Pipeline...")
    pipeline = FeaturePipeline(target_column="Churn")
    X, y = pipeline.fit_transform(df)
    print(f"Engineered Features Matrix Shape: {X.shape}")
    print(f"Sample Derived Features: {[col for col in X.columns if any(k in col for k in ['Ratio', 'Velocity', 'Cohort', 'Index', 'Score'])][:6]}")

    # 4. Model Benchmarking
    print("\n[Step 4/5] Training Benchmark Classifiers...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Logistic Regression Baseline
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    y_prob_lr = lr.predict_proba(X_test)[:, 1]

    # Random Forest Model
    rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    y_prob_rf = rf.predict_proba(X_test)[:, 1]

    # 5. Model Evaluation
    print("\n[Step 5/5] Evaluating Classification Metrics...")
    lr_metrics = ModelEvaluator.evaluate_predictions(y_test, y_pred_lr, y_prob_lr)
    rf_metrics = ModelEvaluator.evaluate_predictions(y_test, y_pred_rf, y_prob_rf)

    print("\n--- Logistic Regression Baseline Results ---")
    print(f"Accuracy : {lr_metrics['accuracy']:.4f}")
    print(f"Precision: {lr_metrics['precision']:.4f}")
    print(f"Recall   : {lr_metrics['recall']:.4f}")
    print(f"F1-Score : {lr_metrics['f1_score']:.4f}")
    print(f"ROC-AUC  : {lr_metrics['roc_auc']:.4f}")
    print(f"PR-AUC   : {lr_metrics['pr_auc']:.4f}")

    print("\n--- Random Forest Classifier Results ---")
    print(f"Accuracy : {rf_metrics['accuracy']:.4f}")
    print(f"Precision: {rf_metrics['precision']:.4f}")
    print(f"Recall   : {rf_metrics['recall']:.4f}")
    print(f"F1-Score : {rf_metrics['f1_score']:.4f}")
    print(f"ROC-AUC  : {rf_metrics['roc_auc']:.4f}")
    print(f"PR-AUC   : {rf_metrics['pr_auc']:.4f}")

    # Feature Importance
    top_features = ModelEvaluator.rank_feature_importance(X.columns.tolist(), rf.feature_importances_, top_n=5)
    print("\nTop 5 Most Predictive Engineered Features (Random Forest):")
    for _, row in top_features.iterrows():
        print(f"  * {row['feature']}: Importance={row['importance']:.4f} (Weight: {row['normalized_weight']:.2%})")

    # Risk Tiering Distribution
    risk_tiers = ModelEvaluator.compute_risk_tiers(y_prob_rf)
    print("\nCustomer Risk Tier Segmentation Breakdown:")
    for tier, count in risk_tiers.value_counts().items():
        print(f"  - {tier}: {count} customers ({count/len(risk_tiers):.1%})")

    print("\n" + "=" * 80)
    print("SUCCESS: Customer Churn EDA & Feature Pipeline Execution Completed.")
    print("=" * 80)


if __name__ == "__main__":
    main()
