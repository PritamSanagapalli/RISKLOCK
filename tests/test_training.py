import pandas as pd
from core import features
from core import train as tr

def test_prepare_training_data_shape_and_columns():
    df = features.prepare_training_data()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty, "Prepared training data should not be empty"

    # Ensure target and features are present
    for col in [*features.NUMERIC_FEATURES, *features.CATEGORICAL_FEATURES, features.TARGET_COL]:
        assert col in df.columns, f"Missing expected column: {col}"


def test_build_pipeline_structure():
    pipeline = tr.build_pipeline()
    # Verify pipeline has expected steps
    assert "preprocessor" in pipeline.named_steps
    assert "model" in pipeline.named_steps

    preprocessor = pipeline.named_steps["preprocessor"]
    # ColumnTransformer exposes transformers_ after fitting, but we can still check type name
    assert preprocessor.__class__.__name__ == "ColumnTransformer"
