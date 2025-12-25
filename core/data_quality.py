import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from .ingestion import load_data
from .features import (
    TARGET_COL,
    create_features,
    filter_financial_data,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    MAX_LOAN_TO_INCOME,
)

# ====================
# CONFIG (POLICY)
# ====================
ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(exist_ok=True)

MAX_MISSING_RATE = 0.10        # 10%
MAX_IMBALANCE_RATIO = 0.85     # 85/15
MAX_DUPLICATES = 0.02          # 2%
MAX_OUTLIERS = 0.12            # 12%

STATUS_PASS = "PASS"
STATUS_FAIL = "FAIL"


def compute_missing_rate(df: pd.DataFrame) -> float:
    return df.isnull().mean().max()


def compute_class_imbalance(df: pd.DataFrame) -> float:
    ratio = df[TARGET_COL].value_counts(normalize=True).max()
    return ratio


def compute_duplicate_rate(df: pd.DataFrame) -> float:
    return df.duplicated().mean()


def compute_outlier_rate(df: pd.DataFrame, numeric_cols) -> float:
    outlier_count = 0
    total = len(df)

    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        outlier_count += outliers

    return outlier_count / (total * len(numeric_cols))


def financial_sanity_checks(df: pd.DataFrame) -> list:
    violations = []

    if (df["annual_inc"] <= 0).any():
        violations.append("Non-positive annual income detected")

    if (df["loan_amnt"] <= 0).any():
        violations.append("Non-positive loan amount detected")

    if (df["loan_amnt"] > df["annual_inc"] * MAX_LOAN_TO_INCOME).any():
        violations.append("Loan amount exceeds policy loan-to-income factor")

    return violations


def compute_quality_score(missing, imbalance, outliers) -> float:
    score = 100
    score -= 2 * (missing * 100)
    score -= 3 * max(0, (imbalance - 0.5) * 100)
    score -= 1 * (outliers * 100)
    return max(0, round(score, 2))


def main():
    df = load_data()

    # Columns required to generate features
    raw_required_columns = {
        "loan_amnt",
        "annual_inc",
        "int_rate",
        "emp_length",
        "dti",
        "earliest_cr_line",
        "term",
        "home_ownership",
        TARGET_COL,
    }

    if not raw_required_columns.issubset(df.columns):
        print(f"Schema mismatch detected. Expected: {raw_required_columns}")
        sys.exit(1)

    # Create features (handles cleaning and derivation)
    df = create_features(df)
    
    # Filter financial data (apply same logic as training)
    df = filter_financial_data(df)
    
    # Subset to Model Columns + Target
    model_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET_COL]
    
    # Ensure all model columns are present (they should be after create_features)
    if not set(model_columns).issubset(df.columns):
        print(f"Feature engineering failed to produce required columns: {set(model_columns) - set(df.columns)}")
        sys.exit(1)

    df = df[model_columns].dropna(subset=[TARGET_COL])

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if TARGET_COL in numeric_cols:
        numeric_cols.remove(TARGET_COL)

    missing_rate = compute_missing_rate(df)
    imbalance = compute_class_imbalance(df)
    duplicate_rate = compute_duplicate_rate(df)
    outlier_rate = compute_outlier_rate(df, numeric_cols)
    finance_violations = financial_sanity_checks(df)

    status = STATUS_PASS
    reasons = []

    if missing_rate > MAX_MISSING_RATE:
        status = STATUS_FAIL
        reasons.append("Missing rate exceeds threshold")

    if imbalance > MAX_IMBALANCE_RATIO:
        status = STATUS_FAIL
        reasons.append("Class imbalance exceeds threshold")

    if duplicate_rate > MAX_DUPLICATES:
        status = STATUS_FAIL
        reasons.append("Duplicate rows exceed threshold")

    if finance_violations:
        status = STATUS_FAIL
        reasons.extend(finance_violations)

    quality_score = compute_quality_score(missing_rate, imbalance, outlier_rate)

    report = {
        "status": status,
        "missing_rate": round(missing_rate, 4),
        "class_imbalance": round(imbalance, 4),
        "duplicate_rate": round(duplicate_rate, 4),
        "outlier_rate": round(outlier_rate, 4),
        "quality_score": quality_score,
        "violations": reasons,
    }

    with open(ARTIFACT_DIR / "data_quality.json", "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))

    if status == STATUS_FAIL:
        sys.exit(1)


if __name__ == "__main__":
    main()
