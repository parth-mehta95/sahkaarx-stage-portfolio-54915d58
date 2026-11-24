"""
Exploratory Data Analysis (EDA) Engine for Customer Churn Intelligence.
Provides automated statistical profiling, outlier detection, and correlation analysis.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd


class EDAEngine:
    """Automated Exploratory Data Analysis engine for structured data science datasets."""

    def __init__(self, target_column: str = "Churn") -> None:
        self.target_column = target_column

    def compute_distribution_summary(
        self, df: pd.DataFrame, numerical_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Computes detailed statistical summary for numerical features:
        mean, std, median, IQR, min, max, skewness, and missing count.
        """
        if numerical_columns is None:
            numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()

        records = []
        for col in numerical_columns:
            series = df[col].dropna()
            q25 = float(series.quantile(0.25)) if not series.empty else 0.0
            q75 = float(series.quantile(0.75)) if not series.empty else 0.0
            iqr = q75 - q25

            records.append(
                {
                    "feature": col,
                    "count": int(series.count()),
                    "missing": int(df[col].isna().sum()),
                    "missing_pct": round(float(df[col].isna().mean() * 100), 2),
                    "mean": round(float(series.mean()), 4) if not series.empty else 0.0,
                    "std": round(float(series.std()), 4) if not series.empty else 0.0,
                    "median": round(float(series.median()), 4) if not series.empty else 0.0,
                    "iqr": round(iqr, 4),
                    "min": round(float(series.min()), 4) if not series.empty else 0.0,
                    "max": round(float(series.max()), 4) if not series.empty else 0.0,
                    "skewness": round(float(series.skew()), 4) if len(series) > 2 else 0.0,
                }
            )

        return pd.DataFrame(records)

    def detect_outliers_iqr(
        self, df: pd.DataFrame, column: str, multiplier: float = 1.5
    ) -> Dict[str, Any]:
        """
        Detects anomalies and outliers in a numeric column using the Interquartile Range (IQR) rule.
        """
        if column not in df.columns:
            raise KeyError(f"Column '{column}' not found in dataframe.")

        series = df[column].dropna()
        q25 = float(series.quantile(0.25))
        q75 = float(series.quantile(0.75))
        iqr = q75 - q25
        lower_bound = q25 - (multiplier * iqr)
        upper_bound = q75 + (multiplier * iqr)

        outlier_mask = (df[column] < lower_bound) | (df[column] > upper_bound)
        outlier_indices = df[outlier_mask].index.tolist()

        return {
            "feature": column,
            "q25": round(q25, 4),
            "q75": round(q75, 4),
            "iqr": round(iqr, 4),
            "lower_bound": round(lower_bound, 4),
            "upper_bound": round(upper_bound, 4),
            "outlier_count": len(outlier_indices),
            "outlier_percentage": round((len(outlier_indices) / len(df)) * 100, 2),
            "outlier_indices": outlier_indices,
        }

    def compute_correlations(
        self, df: pd.DataFrame, target_col: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Calculates Pearson and Spearman correlations between numerical features
        and the target variable.
        """
        target = target_col or self.target_column
        numeric_df = df.select_dtypes(include=[np.number]).copy()

        if target in df.columns and target not in numeric_df.columns:
            # Convert binary target to integer if string ('Yes'/'No', '1'/'0')
            target_series = df[target].map({"Yes": 1, "No": 0, "1": 1, "0": 0})
            if target_series.isna().any():
                target_series = pd.factorize(df[target])[0]
            numeric_df[target] = target_series

        if target not in numeric_df.columns:
            return pd.DataFrame()

        pearson_corr = numeric_df.corr(method="pearson")[target].drop(target)
        spearman_corr = numeric_df.corr(method="spearman")[target].drop(target)

        corr_df = pd.DataFrame(
            {
                "feature": pearson_corr.index,
                "pearson_correlation": pearson_corr.values.round(4),
                "spearman_correlation": spearman_corr.values.round(4),
                "abs_pearson": pearson_corr.abs().values.round(4),
            }
        ).sort_values(by="abs_pearson", ascending=False).reset_index(drop=True)

        return corr_df

    def generate_eda_profile(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generates an end-to-end exploratory data analysis summary profile.
        """
        total_rows, total_cols = df.shape
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

        class_balance = {}
        if self.target_column in df.columns:
            counts = df[self.target_column].value_counts()
            class_balance = {
                str(k): {"count": int(v), "ratio": round(float(v / total_rows), 4)}
                for k, v in counts.items()
            }

        distribution_df = self.compute_distribution_summary(df, numerical_cols)
        correlation_df = self.compute_correlations(df)

        return {
            "total_records": total_rows,
            "total_features": total_cols,
            "numerical_features_count": len(numerical_cols),
            "categorical_features_count": len(categorical_cols),
            "numerical_features": numerical_cols,
            "categorical_features": categorical_cols,
            "class_distribution": class_balance,
            "distribution_summary": distribution_df.to_dict(orient="records"),
            "top_correlations": correlation_df.head(5).to_dict(orient="records"),
        }
