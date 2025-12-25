import json
from pathlib import Path

import pandas as pd
from joblib import load
from core.features import create_features

# Use absolute path based on this file's location
BASE_DIR = Path(__file__).parent.parent
ARTIFACT_DIR = BASE_DIR / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "model.joblib"
DATA_QUALITY_PATH = ARTIFACT_DIR / "data_quality.json"
TRAIN_METADATA_PATH = ARTIFACT_DIR / "train_metadata.json"

# Load artifacts once at startup
MODEL = load(MODEL_PATH)

# Load data quality with proper error handling
try:
    if DATA_QUALITY_PATH.exists():
        with open(DATA_QUALITY_PATH) as f:
            content = f.read().strip()
            if content:
                DATA_QUALITY = json.loads(content)
            else:
                DATA_QUALITY = {"quality_score": None}
    else:
        DATA_QUALITY = {"quality_score": None}
except (json.JSONDecodeError, Exception):
    DATA_QUALITY = {"quality_score": None}  # Default value if file is empty, missing, or invalid

# Load train metadata with error handling
try:
    with open(TRAIN_METADATA_PATH) as f:
        TRAIN_METADATA = json.load(f)
except (json.JSONDecodeError, FileNotFoundError):
    TRAIN_METADATA = {"model_type": "v1"}

def risk_band(prob: float) -> str:
    if prob >= 0.7:
        return "High"
    if prob >= 0.4:
        return "Medium"
    return "Low"

def explain(model, X: pd.DataFrame) -> list[str]:
    try:
        # Check if the model is a Pipeline
        if hasattr(model, "named_steps"):
            # Access the classifier
            classifier = model.named_steps["model"]
            # Access the preprocessor
            preprocessor = model.named_steps["preprocessor"]
            
            # Get feature names from preprocessor
            feature_names = preprocessor.get_feature_names_out()
            
            # Get coefficients from classifier
            if hasattr(classifier, "coef_"):
                coefs = classifier.coef_[0]
                
                # Map coefficients to feature names
                contributions = dict(zip(feature_names, coefs))
                top = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
                return [name for name, _ in top]
    except Exception:
        pass
    return []

def predict_loan(payload: dict) -> dict:
    X = pd.DataFrame([payload])
    
    # Apply feature engineering (same as training)
    X = create_features(X)
    
    prob = float(MODEL.predict_proba(X)[0, 1])

    return {
        "probability_default": round(prob, 4),
        "risk_level": risk_band(prob),
        "top_risk_factors": explain(MODEL, X),
        "model_version": TRAIN_METADATA.get("model_type", "v1"),
        "data_quality_score": DATA_QUALITY.get("quality_score"),
    }
