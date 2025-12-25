import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import roc_auc_score

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
MODEL_PATH = ARTIFACT_DIR / "model.joblib"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"

# Conservative finance thresholds
MIN_ROC_AUC = 0.695  # policy tolerance band
MIN_KS_STAT = 0.25


# ====================
# METRICS
# ====================
def ks_statistic(y_true, y_prob) -> float:
    """
    Kolmogorov–Smirnov statistic for binary classification.
    """
    df = pd.DataFrame({"y": y_true, "p": y_prob})
    df = df.sort_values("p")

    df["cum_pos"] = (df["y"] == 1).cumsum() / (df["y"] == 1).sum()
    df["cum_neg"] = (df["y"] == 0).cumsum() / (df["y"] == 0).sum()

    ks = np.max(np.abs(df["cum_pos"] - df["cum_neg"]))
    return float(ks)


# ====================
# EVALUATION
# ====================
def main():
    if not MODEL_PATH.exists():
        print("Model artifact not found. Train the model first.")
        sys.exit(1)

    # Load model
    model = load(MODEL_PATH)

    # Load governed dataset (same policy as training)
    df = prepare_training_data()

    if df.empty:
        print("Evaluation aborted: no data available")
        sys.exit(1)

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET_COL]

    # Predict probabilities
    y_prob = model.predict_proba(X)[:, 1]

    # Metrics
    roc_auc = roc_auc_score(y, y_prob)
    ks = ks_statistic(y, y_prob)

    status = "PASS"
    reasons = []

    if roc_auc < MIN_ROC_AUC:
        status = "FAIL"
        reasons.append(
            f"ROC-AUC {roc_auc:.3f} below threshold {MIN_ROC_AUC}"
        )

    if ks < MIN_KS_STAT:
        status = "FAIL"
        reasons.append(
            f"KS statistic {ks:.3f} below threshold {MIN_KS_STAT}"
        )

    metrics = {
        "status": status,
        "roc_auc": round(roc_auc, 4),
        "ks_statistic": round(ks, 4),
        "thresholds": {
            "min_roc_auc": MIN_ROC_AUC,
            "min_ks_statistic": MIN_KS_STAT,
        },
        "violations": reasons,
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))

    if status == "FAIL":
        sys.exit(1)


if __name__ == "__main__":
    main()
