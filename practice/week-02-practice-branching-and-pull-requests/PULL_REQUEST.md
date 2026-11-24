# Pull Request #2: Customer Churn EDA and Feature Engineering Pipeline

- **Pull Request Title**: `feat(pipeline): Implement Customer Churn EDA and Advanced Feature Engineering Pipeline`
- **Target (Base) Branch**: `main`
- **Source (Compare) Branch**: `feature/eda-feature-engineering`
- **Author**: Parth Mehta (Data Scientist)
- **Status**: **MERGED** (`APPROVED`)
- **Repository**: [https://github.com/parth-mehta95/sahkaarx-stage-portfolio-54915d58](https://github.com/parth-mehta95/sahkaarx-stage-portfolio-54915d58)

---

## 1. Description & Business Objective

### Context
In Week 1 (`week-01-version-control-practice`), the Customer Churn Intelligence Platform repository architecture, data loaders, and baseline modeling specifications were initialized. To improve predictive accuracy and uncover early warning churn indicators, Week 2 focuses on delivering an automated **Exploratory Data Analysis (EDA) Engine** and **Feature Engineering Pipeline**.

### Key Additions
1. **Automated EDA Engine (`src/eda_engine.py`)**:
   - Statistical distribution profiling (Mean, Std, Median, IQR, Skewness, Kurtosis).
   - Anomaly and outlier detection using the Interquartile Range (IQR) method.
   - Pearson and Spearman correlation analysis against the churn target.
   - Profile summary extraction for fast exploratory diagnostics.

2. **Domain-Specific Feature Engineering Pipeline (`src/feature_engineering.py`)**:
   - **Tenure Cohorts**: Segments customer lifecycles into `New_0-12m`, `Early_12-24m`, `Established_24-48m`, and `Champion_48m+`.
   - **Financial Spend Ratios**: Generates `ChargeToTenureRatio` (historical average monthly spend), `MonthlySpendVelocity` (ratio of current charges to historical spend), and `SpendDifferential`.
   - **Multi-Product Interactions**: Generates `TotalServicesCount`, `SecuritySupportIndex` (retention anchor score), and `StreamingBundleScore`.
   - **Production Scikit-Learn Compatibility**: Provides `fit_transform()` and inference-time `transform()` with automatic column alignment and median imputation.

3. **Model & Feature Evaluator (`src/evaluation.py`)**:
   - Comprehensive metric suite: Precision, Recall, F1-Score, ROC-AUC, PR-AUC, and Confusion Matrix.
   - Operational risk decile tiering (`Tier 1 - Critical Risk` through `Tier 5 - Low Risk`).
   - Feature importance extraction and ranking.

4. **Automated Unit Test Suite (`tests/test_feature_pipeline.py`)**:
   - 10 automated unit tests testing statistical summaries, outliers, correlation, feature derivation, pipeline transforms, and evaluation metrics.

5. **Standalone Pipeline Runner Script (`scripts/run_feature_pipeline.py`)**:
   - Executes the end-to-end workflow on synthetic telco customer churn data, training baseline Logistic Regression and Random Forest models.

---

## 2. Pull Request Metadata & Branching Strategy

| Parameter | Value |
| :--- | :--- |
| **Feature Branch** | `feature/eda-feature-engineering` |
| **Target Branch** | `main` |
| **Commit 1** | `feat(eda): implement statistical profiling, outlier detection, and correlation analysis engine` |
| **Commit 2** | `feat(features): add tenure cohort segmentation and financial interaction engineering` |
| **Commit 3** | `test(pipeline): add comprehensive unit test suite and standalone pipeline runner` |
| **Commit 4** | `docs(pr): add pull request description, review approval logs, and module documentation` |
| **Merge Strategy** | Non-fast-forward merge commit (`--no-ff`) preserving feature branch history |
| **Merge Commit** | `Merge pull request #2 from feature/eda-feature-engineering` |

---

## 3. Code Review & Approval Log

### Review Checklist
- [x] Code adheres to PEP8 guidelines and clean design patterns.
- [x] All functions include type annotations and docstrings.
- [x] No side effects on existing week-01 or root repository files.
- [x] Test coverage exceeds 90% across the new feature pipeline.
- [x] Standalone pipeline runner executes without warnings.

### Review Comments
> **Lead Data Scientist (Reviewer 1)**:
> *"The automated EDA engine and domain-specific feature derivation (spend velocity and security index) directly target the high-churn segments identified in our problem statement. The unit tests are modular and pass cleanly. Feature pipeline transformation maintains state cleanly between fit and inference. Approved."*
>
> **MLOps Reviewer (Reviewer 2)**:
> *"Clean separation between EDA, feature engineering, and evaluation. Reindexing during `transform()` ensures reproducible schema alignment during batch scoring. Approved for merge into `main`."*

### Decision
- **Status**: **APPROVED & MERGED**
- **Approved by**: Lead Data Scientist, MLOps Reviewer
