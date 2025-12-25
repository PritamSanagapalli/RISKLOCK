from pathlib import Path
import pandas as pd

# ====================
# CONFIG
# ====================
DATA_PATH = Path("Data/Raw/loan.csv")
SAMPLE_PATH = Path("Data/Raw/loan_sample.csv")

def load_data() -> pd.DataFrame:
    """
    Loads the raw dataset from CSV.
    Prioritizes the full dataset, falls back to sample for CI/CD portability.
    """
    path = DATA_PATH if DATA_PATH.exists() else SAMPLE_PATH

    if not path.exists():
        raise FileNotFoundError(f"Neither full dataset ({DATA_PATH}) nor sample ({SAMPLE_PATH}) found.")

    print(f"Loading data from: {path}")
    # Reduce mixed-type DtypeWarning by disabling low-memory chunked inference
    df = pd.read_csv(path, low_memory=False)
    return df
