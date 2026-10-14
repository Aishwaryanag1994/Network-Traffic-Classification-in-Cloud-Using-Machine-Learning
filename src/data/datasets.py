import pandas as pd
import numpy as np
from pathlib import Path
from .utils import drop_na_and_duplicates

def load_cicids2017_csvs(data_dir: Path) -> pd.DataFrame:
    data_dir = Path(data_dir)
    csvs = sorted(data_dir.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV files found under {data_dir}")
    frames = []
    for p in csvs:
        try:
            df = pd.read_csv(p, low_memory=False)
            frames.append(df)
        except Exception as e:
            print(f"Skipping {p}: {e}")
    return pd.concat(frames, axis=0, ignore_index=True)

def prepare_dataset(df: pd.DataFrame, label_col: str, id_cols=None):
    if id_cols is None: id_cols = []
    df = df.copy()
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found")
    df[label_col] = df[label_col].astype(str).str.strip()
    df = drop_na_and_duplicates(df)
    df = df.replace([np.inf, -np.inf], np.nan)
    y = df[label_col]
    X = df.drop(columns=[c for c in id_cols if c in df.columns] + [label_col])
    X = X.select_dtypes(include=[np.number])
    return X, y, df
