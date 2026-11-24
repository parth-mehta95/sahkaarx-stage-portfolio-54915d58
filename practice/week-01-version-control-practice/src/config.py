"""
Configuration module for the Data Science Capstone Project.
Handles project paths, parameters, and environment settings.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List


@dataclass
class ProjectPaths:
    """Project directory paths."""
    ROOT_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = field(default=None)
    RAW_DATA_DIR: Path = field(default=None)
    PROCESSED_DATA_DIR: Path = field(default=None)
    MODELS_DIR: Path = field(default=None)
    LOGS_DIR: Path = field(default=None)

    def __post_init__(self):
        self.DATA_DIR = self.ROOT_DIR / "data"
        self.RAW_DATA_DIR = self.DATA_DIR / "raw"
        self.PROCESSED_DATA_DIR = self.DATA_DIR / "processed"
        self.MODELS_DIR = self.ROOT_DIR / "models"
        self.LOGS_DIR = self.ROOT_DIR / "logs"


@dataclass
class ModelConfig:
    """Model hyperparameters and training configuration."""
    RANDOM_STATE: int = 42
    TEST_SIZE: float = 0.20
    CV_FOLDS: int = 5
    TARGET_COLUMN: str = "Churn"
    CATEGORICAL_FEATURES: List[str] = field(default_factory=lambda: [
        "gender", "Partner", "Dependents", "PhoneService", "InternetService",
        "Contract", "PaperlessBilling", "PaymentMethod"
    ])
    NUMERICAL_FEATURES: List[str] = field(default_factory=lambda: [
        "SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"
    ])


paths = ProjectPaths()
model_config = ModelConfig()
