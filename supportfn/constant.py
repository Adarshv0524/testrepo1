import os
import pandas as pd

# Path configuration
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")  # Points to testrepo1/data

def read_csv(file_name: str) -> pd.DataFrame:
    """Read CSV file from data directory"""
    file_path = os.path.join(DATA_DIR, file_name)
    return pd.read_csv(file_path)