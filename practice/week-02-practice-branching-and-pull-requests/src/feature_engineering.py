"""
Feature Engineering Pipeline for Customer Churn Intelligence.
Implements domain-specific feature derivation, tenure cohort segmentation,
spend velocity metrics, service interaction synthesis, and robust scaling.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class FeaturePipeline:
    """
    End-to-end feature transformation pipeline for customer churn data.
    Designed with production-grade scikit-learn compatibility.
    """

    SERVICE_COLUMNS = [
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]

    SECURITY_SERVICES = ["OnlineSecurity", "DeviceProtection", "TechSupport"]

    def __init__(
        self,
        target_column: str = "Churn",
        tenure_column: str = "tenure",
        monthly_charges_col: str = "MonthlyCharges",
        total_charges_col: str = "TotalCharges",
    ) -> None:
        self.target_column = target_column
        self.tenure_col = tenure_column
        self.monthly_col = monthly_charges_col
        self.total_col = total_charges_col
        self.scaler = StandardScaler()
        self.fitted = False
        self.feature_names_: List[str] = []
        self.median_values_: Dict[str, float] = {}

    def create_tenure_cohorts(self, df: pd.DataFrame) -> pd.Series:
        """
        Segments customers into standard lifecycle tenure cohorts:
        - 0-12m: New onboarding (highest risk)
        - 12-24m: Early adoption
        - 24-48m: Established accounts
        - 48m+: Champion / Loyal long-term
        """
        if self.tenure_col not in df.columns:
            raise KeyError(f"Tenure column '{self.tenure_col}' missing.")

        bins = [-1, 12, 24, 48, np.inf]
        labels = ["New_0-12m", "Early_12-24m", "Established_24-48m", "Champion_48m+"]
        cohorts = pd.cut(df[self.tenure_col], bins=bins, labels=labels)
        return cohorts.astype(str)

    def engineer_financial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Synthesizes behavioral financial ratios:
        - ChargeToTenureRatio: historical average monthly spend
        - MonthlySpendVelocity: ratio of current monthly charges to historical monthly average
        - SpendDifferential: divergence between expected spend and actual monthly charge
        """
        df_out = pd.DataFrame(index=df.index)

        # Ensure numeric conversion
        total = pd.to_numeric(df[self.total_col], errors="coerce").fillna(0.0)
        monthly = pd.to_numeric(df[self.monthly_col], errors="coerce").fillna(0.0)
        tenure = pd.to_numeric(df[self.tenure_col], errors="coerce").fillna(0.0)

        # Average historical monthly charges
        charge_to_tenure = total / (tenure + 1.0)
        df_out["ChargeToTenureRatio"] = charge_to_tenure.round(4)

        # Spend velocity indicates recent upselling or price hikes
        spend_velocity = monthly / (charge_to_tenure + 1e-5)
        df_out["MonthlySpendVelocity"] = spend_velocity.round(4)

        # Financial burden differential
        expected_total = monthly * tenure
        df_out["SpendDifferential"] = (total - expected_total).round(4)

        return df_out

    def engineer_service_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Constructs composite scores from multi-product adoption:
        - TotalServicesCount: breadth of customer engagement
        - SecuritySupportIndex: security and support bundle adoption (strong retention driver)
        - StreamingBundleScore: entertainment engagement index
        """
        df_out = pd.DataFrame(index=df.index)

        # Count active services
        active_counts = np.zeros(len(df))
        for col in self.SERVICE_COLUMNS:
            if col in df.columns:
                active_counts += (
                    df[col].isin(["Yes", "Fiber optic", "DSL"]).astype(int).values
                )
        df_out["TotalServicesCount"] = active_counts

        # Security and tech support composite index
        sec_counts = np.zeros(len(df))
        for col in self.SECURITY_SERVICES:
            if col in df.columns:
                sec_counts += (df[col] == "Yes").astype(int).values
        df_out["SecuritySupportIndex"] = sec_counts

        # Streaming composite index
        streaming_cols = ["StreamingTV", "StreamingMovies"]
        stream_counts = np.zeros(len(df))
        for col in streaming_cols:
            if col in df.columns:
                stream_counts += (df[col] == "Yes").astype(int).values
        df_out["StreamingBundleScore"] = stream_counts

        return df_out

    def fit_transform(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Fits the preprocessing and transformation pipeline on training data,
        deriving new features, encoding categories, and standardizing numeric features.
        """
        df_copy = df.copy()

        # Extract target if present
        target_series = None
        if self.target_column in df_copy.columns:
            target_raw = df_copy.pop(self.target_column)
            target_series = target_raw.map({"Yes": 1, "No": 0, "1": 1, "0": 0})
            if target_series.isna().any():
                target_series = pd.Series(pd.factorize(target_raw)[0], index=df.index)

        # Store medians for imputation
        numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
        for c in numeric_cols:
            self.median_values_[c] = float(df_copy[c].median())

        # Derive features
        df_copy["TenureCohort"] = self.create_tenure_cohorts(df_copy)
        fin_features = self.engineer_financial_features(df_copy)
        svc_features = self.engineer_service_interactions(df_copy)

        combined_df = pd.concat([df_copy, fin_features, svc_features], axis=1)

        # One-hot encode categorical features
        cat_cols = combined_df.select_dtypes(exclude=[np.number]).columns.tolist()
        encoded_df = pd.get_dummies(combined_df, columns=cat_cols, drop_first=True)

        self.feature_names_ = encoded_df.columns.tolist()

        # Scale features
        scaled_array = self.scaler.fit_transform(encoded_df)
        X_scaled = pd.DataFrame(scaled_array, columns=self.feature_names_, index=df.index)

        self.fitted = True
        return X_scaled, target_series

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms new incoming inference batches using fitted statistics.
        """
        if not self.fitted:
            raise RuntimeError("Pipeline must be fitted before calling transform().")

        df_copy = df.copy()
        if self.target_column in df_copy.columns:
            df_copy.pop(self.target_column)

        # Impute missing values with learned medians
        for col, med in self.median_values_.items():
            if col in df_copy.columns:
                df_copy[col] = df_copy[col].fillna(med)

        df_copy["TenureCohort"] = self.create_tenure_cohorts(df_copy)
        fin_features = self.engineer_financial_features(df_copy)
        svc_features = self.engineer_service_interactions(df_copy)

        combined_df = pd.concat([df_copy, fin_features, svc_features], axis=1)

        # Categorical encoding aligned to learned feature set
        cat_cols = combined_df.select_dtypes(exclude=[np.number]).columns.tolist()
        encoded_df = pd.get_dummies(combined_df, columns=cat_cols, drop_first=True)

        # Reindex to ensure identical feature columns
        aligned_df = encoded_df.reindex(columns=self.feature_names_, fill_value=0.0)

        scaled_array = self.scaler.transform(aligned_df)
        return pd.DataFrame(scaled_array, columns=self.feature_names_, index=df.index)
