import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from joblib import dump

from .features import (
    prepare_training_data,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COL,
)

# ====================
# CONFIG
# ====================
ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(exist_ok=True)

MODEL_PATH = ARTIFACT_DIR / "model.joblib"
METADATA_PATH = ARTIFACT_DIR / "train_metadata.json"

RANDOM_STATE = 42
TEST_SIZE = 0.2

# Metric gate (soft – hard gate happens in evaluate.py)
MIN_TRAIN_ROC_AUC = 0.65


# ====================
# MODEL PIPELINE
# ====================
def build_pipeline() -> Pipeline:
    numeric_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=RANDOM_STATE,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    return pipeline


# ====================
# TRAINING
# ====================
def main():
    df = prepare_training_data()

    if df.empty:
        print("Training aborted: no valid rows after filtering")
        sys.exit(1)

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET_COL]

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    # Quick sanity metric (final gating happens in evaluate.py)
    val_probs = pipeline.predict_proba(X_val)[:, 1]
    roc_auc = roc_auc_score(y_val, val_probs)

    if roc_auc < MIN_TRAIN_ROC_AUC:
        print(
            f"Training aborted: ROC-AUC {roc_auc:.3f} below minimum {MIN_TRAIN_ROC_AUC}"
        )
        sys.exit(1)

    # Persist model
    dump(pipeline, MODEL_PATH)

    metadata = {
        "model_type": "LogisticRegression",
        "roc_auc_validation": round(float(roc_auc), 4),
        "rows_used": len(df),
        "features": {
            "numeric": NUMERIC_FEATURES,
            "categorical": CATEGORICAL_FEATURES,
        },
        "random_state": RANDOM_STATE,
    }

    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    print("Training completed successfully")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
