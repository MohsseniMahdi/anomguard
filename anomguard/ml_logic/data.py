import pandas as pd
from pathlib import Path

def load_local_data(cache_path: Path) -> pd.DataFrame:
    """
    Load data from a local CSV file.
    """
    try:
        df = pd.read_csv(cache_path)
        print(f"✅ Data loaded from {cache_path}, with shape {df.shape}")
        return df
    except FileNotFoundError:
        print(f"❌ File not found at {cache_path}")
        return None
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None
