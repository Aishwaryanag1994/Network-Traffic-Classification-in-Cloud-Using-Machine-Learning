import numpy as np
import pandas as pd

def entropy(arr):
    vals, counts = np.unique(arr[~np.isnan(arr)], return_counts=True)
    total = counts.sum()
    if total == 0:
        return 0.0
    probs = counts / total
    return float(-(probs * np.log2(probs + 1e-12)).sum())

def add_engineered_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    len_cols = [c for c in X.columns if "Packet Length" in c or "Length" in c]
    if len_cols:
        X["LenEntropy"] = X[len_cols].apply(lambda r: entropy(r.values.astype(float)), axis=1)
    else:
        X["LenEntropy"] = 0.0
    if "Fwd IAT Mean" in X.columns and "Fwd IAT Std" in X.columns:
        den = X["Fwd IAT Mean"].replace(0, np.nan)
        X["IATCoeffVar"] = (X["Fwd IAT Std"] / den).fillna(0.0)
    else:
        X["IATCoeffVar"] = 0.0
    num = X["Total Length of Fwd Packets"] if "Total Length of Fwd Packets" in X.columns else 0.0
    den = X["Total Length of Bwd Packets"] if "Total Length of Bwd Packets" in X.columns else 1.0
    try:
        X["ByteRateRatio"] = (num / den.replace(0, np.nan)).fillna(0.0)
    except Exception:
        X["ByteRateRatio"] = 0.0
    return X
