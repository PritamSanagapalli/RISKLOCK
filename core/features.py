import pandas as pd
import numpy as np
from .ingestion import load_data

# ====================
# FEATURE DEFINITION
# ====================
TARGET_COL = "loan_status"

NUMERIC_FEATURES = [
    "loan_amnt",
    "annual_inc",
    "int_rate",
    "emp_length",
    "dti",
    "credit_hist_length",
]

CATEGORICAL_FEATURES = [
    "term",
    "home_ownership",
]

MAX_LOAN_TO_INCOME = 10

# ====================
# DATA PREPARATION
# ====================
def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies feature engineering and cleaning to the dataframe.
    """
    df = df.copy()

    # Feature cleaning
    if "int_rate" in df.columns:
        df["int_rate"] = (
            df["int_rate"].astype(str).str.replace("%", "").astype(float)
        )

    if "emp_length" in df.columns:
        df["emp_length"] = (
            df["emp_length"]
            .astype(str)
            .str.replace("years", "", regex=False)
            .str.replace("year", "", regex=False)
            .str.replace("+", "", regex=False)
            .str.replace("<", "", regex=False)
            .str.strip()
            .replace("n/a", np.nan)
            .astype(float)
            .fillna(0)
        )

    if "earliest_cr_line" in df.columns:
        df["earliest_cr_line"] = pd.to_datetime(
            df["earliest_cr_line"], format="%b-%Y", errors="coerce"
        )
        today = pd.Timestamp.today()
        df["credit_hist_length"] = (
            (today - df["earliest_cr_line"]).dt.days / 365
        ).fillna(0)

    return df


def filter_financial_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters out rows that violate financial sanity checks.
    """
    if "annual_inc" in df.columns:
        df = df[df["annual_inc"] > 0]
    
    if "loan_amnt" in df.columns:
        df = df[df["loan_amnt"] > 0]
        
    if "loan_amnt" in df.columns and "annual_inc" in df.columns:
        df = df[df["loan_amnt"] <= df["annual_inc"] * MAX_LOAN_TO_INCOME]
        
    return df


def prepare_training_data() -> pd.DataFrame:
    """
    Loads data using the ingestion module and applies feature engineering.
    """
    df = load_data()

    # Apply feature engineering
    df = create_features(df)
    
    # Apply financial filtering
    df = filter_financial_data(df)

    # Keep only required columns
    required_columns = (
        NUMERIC_FEATURES
        + CATEGORICAL_FEATURES
        + [TARGET_COL]
    )

    df = df[[col for col in required_columns if col in df.columns]]

    # Target policy
    df = df[df[TARGET_COL].isin(["Fully Paid", "Charged Off", "Default"])]
    df[TARGET_COL] = df[TARGET_COL].map(
        {"Fully Paid": 0, "Charged Off": 1, "Default": 1}
    ).astype(int)

    df = df.dropna()

    return df
