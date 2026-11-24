"""
Utilities module: logging, metric formatting, and helpers.
"""

import logging
import sys
from typing import Dict, Any


def setup_logger(name: str = "capstone_logger", level: int = logging.INFO) -> logging.Logger:
    """Set up and configure a standard logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def format_metrics(metrics: Dict[str, float]) -> str:
    """Format evaluation metrics as a readable string."""
    return " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
