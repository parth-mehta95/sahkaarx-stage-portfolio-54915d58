"""
Unit tests to verify project initialization, configuration, data loading,
and baseline model functionality.
"""

import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import paths, model_config
from src.data_loader import generate_synthetic_data, validate_schema, preprocess_data
from src.model import ChurnModelPipeline


def test_paths_configuration():
    """Verify project paths are correctly resolved."""
    assert paths.ROOT_DIR.exists(), "Root directory should exist."
    assert paths.DATA_DIR.name == "data"
    assert paths.MODELS_DIR.name == "models"


def test_model_configuration():
    """Verify model hyperparameters and feature columns are set."""
    assert model_config.TARGET_COLUMN == "Churn"
    assert model_config.RANDOM_STATE == 42
    assert len(model_config.CATEGORICAL_FEATURES) > 0
    assert len(model_config.NUMERICAL_FEATURES) > 0


def test_synthetic_data_generation():
    """Verify synthetic data generator produces valid dataframe."""
    df = generate_synthetic_data(n_samples=50, random_state=42)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 50
    assert "customerID" in df.columns
    assert "Churn" in df.columns
    assert validate_schema(df) is True


def test_data_preprocessing():
    """Verify data preprocessing pipeline splits features and target correctly."""
    df = generate_synthetic_data(n_samples=60, random_state=42)
    X, y = preprocess_data(df)
    assert len(X) == 60
    assert len(y) == 60
    assert set(np.unique(y)).issubset({0, 1})
    assert X.shape[1] > 0


def test_baseline_model_pipeline():
    """Verify model training and evaluation pipeline on synthetic data."""
    df = generate_synthetic_data(n_samples=100, random_state=42)
    X, y = preprocess_data(df)
    
    # Train / test split
    X_train, X_test = X.iloc[:80], X.iloc[80:]
    y_train, y_test = y.iloc[:80], y.iloc[80:]
    
    pipeline = ChurnModelPipeline(model_type="random_forest")
    pipeline.train(X_train, y_train)
    assert pipeline.is_fitted is True
    
    metrics = pipeline.evaluate(X_test, y_test)
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1_score" in metrics
    assert "roc_auc" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0
