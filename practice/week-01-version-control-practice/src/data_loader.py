"""
Data loader and preprocessing module for Customer Churn dataset.
"""

from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
from src.config import model_config, paths
from src.utils import setup_logger

logger = setup_logger("data_loader")


def generate_synthetic_data(n_samples: int = 200, random_state: int = 42) -> pd.DataFrame:
    """Generate synthetic customer churn dataset for initial testing."""
    rng = np.random.default_rng(random_state)
    
    genders = rng.choice(["Male", "Female"], size=n_samples)
    senior_citizen = rng.choice([0, 1], size=n_samples, p=[0.84, 0.16])
    partner = rng.choice(["Yes", "No"], size=n_samples)
    dependents = rng.choice(["Yes", "No"], size=n_samples, p=[0.3, 0.7])
    tenure = rng.integers(1, 73, size=n_samples)
    phone_service = rng.choice(["Yes", "No"], size=n_samples, p=[0.9, 0.1])
    internet_service = rng.choice(["DSL", "Fiber optic", "No"], size=n_samples)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], size=n_samples)
    paperless_billing = rng.choice(["Yes", "No"], size=n_samples)
    payment_method = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
        size=n_samples
    )
    monthly_charges = np.round(rng.uniform(18.25, 118.75, size=n_samples), 2)
    total_charges = np.round(monthly_charges * tenure + rng.uniform(-10, 10, size=n_samples), 2)
    total_charges = np.maximum(total_charges, monthly_charges)
    
    # Calculate churn probability based on features
    churn_score = (
        (contract == "Month-to-month") * 1.5 +
        (tenure < 12) * 1.0 +
        (monthly_charges > 70) * 0.8 +
        rng.normal(0, 0.5, size=n_samples)
    )
    churn = np.where(churn_score > 1.5, "Yes", "No")

    df = pd.DataFrame({
        "customerID": [f"CUST-{i:05d}" for i in range(1, n_samples + 1)],
        "gender": genders,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "InternetService": internet_service,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "Churn": churn
    })
    return df


def validate_schema(df: pd.DataFrame) -> bool:
    """Validate that required features and target exist in DataFrame."""
    required = set(model_config.CATEGORICAL_FEATURES + model_config.NUMERICAL_FEATURES + [model_config.TARGET_COLUMN])
    missing = required - set(df.columns)
    if missing:
        logger.error(f"Missing required columns in dataset: {missing}")
        return False
    logger.info(f"Schema validated successfully. Dataset shape: {df.shape}")
    return True


def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Clean and preprocess raw dataframe into features (X) and target (y)."""
    df = df.copy()
    # Convert TotalCharges to numeric handling any blanks
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    
    # Encode target
    y = (df[model_config.TARGET_COLUMN] == "Yes").astype(int)
    
    # Prepare features
    feature_cols = model_config.NUMERICAL_FEATURES + model_config.CATEGORICAL_FEATURES
    X = df[feature_cols]
    
    # One-hot encode categorical features
    X = pd.get_dummies(X, columns=model_config.CATEGORICAL_FEATURES, drop_first=True)
    return X, y
