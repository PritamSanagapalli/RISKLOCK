import json
from pathlib import Path

import pytest

from core import data_quality as dq


ARTIFACT_PATH = Path("artifacts/data_quality.json")


def test_data_quality_artifact_generated():
    # Run data quality gate; it may exit(1) on policy violations, which is acceptable for this test
    try:
        dq.main()
    except SystemExit:
        pass

    assert ARTIFACT_PATH.exists(), "data_quality.json artifact should be created"

    content = ARTIFACT_PATH.read_text().strip()
    assert content, "data_quality.json should not be empty"
    report = json.loads(content)

    # Validate required keys exist in the report regardless of PASS/FAIL status
    for key in [
        "status",
        "missing_rate",
        "class_imbalance",
        "duplicate_rate",
        "outlier_rate",
        "quality_score",
        "violations",
    ]:
        assert key in report, f"Missing key in data quality report: {key}"

    assert isinstance(report["violations"], list)
    assert 0 <= float(report["quality_score"]) <= 100