
import argparse, joblib
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest

ALIASES = {
    "Flow Duration": ["Flow Duration"],
    "Total Fwd Packets": ["Total Fwd Packets", "Tot Fwd Pkts"],
    "Total Backward Packets": ["Total Backward Packets", "Tot Bwd Pkts"],
    "Total Length of Fwd Packets": ["Total Length of Fwd Packets", "TotLen Fwd Pkts"],
    "Total Length of Bwd Packets": ["Total Length of Bwd Packets", "TotLen Bwd Pkts"],
    "Fwd Packet Length Max": ["Fwd Packet Length Max", "Fwd Pkt Len Max"],
    "Fwd Packet Length Min": ["Fwd Packet Length Min", "Fwd Pkt Len Min"],
    "Fwd Packet Length Mean": ["Fwd Packet Length Mean", "Fwd Pkt Len Mean"],
    "Bwd Packet Length Max": ["Bwd Packet Length Max", "Bwd Pkt Len Max"],
    "Bwd Packet Length Min": ["Bwd Packet Length Min", "Bwd Pkt Len Min"],
    "Bwd Packet Length Mean": ["Bwd Packet Length Mean", "Bwd Pkt Len Mean"],
    "Flow Bytes/s": ["Flow Bytes/s", "Flow Byts/s"],
    "Flow Packets/s": ["Flow Packets/s", "Flow Pkts/s"],
    "Down/Up Ratio": ["Down/Up Ratio"],
    "Avg Packet Size": ["Avg Packet Size", "Pkt Size Avg"],
    "Init_Win_bytes_forward": ["Init_Win_bytes_forward", "Init Fwd Win Byts"],
    "Init_Win_bytes_backward": ["Init_Win_bytes_backward", "Init Bwd Win Byts"],
    "Fwd IAT Mean": ["Fwd IAT Mean"], "Fwd IAT Std": ["Fwd IAT Std"],
    "Bwd IAT Mean": ["Bwd IAT Mean"], "Bwd IAT Std": ["Bwd IAT Std"],
    "Subflow Fwd Bytes": ["Subflow Fwd Bytes", "Subflow Fwd Byts"],
    "Subflow Bwd Bytes": ["Subflow Bwd Bytes", "Subflow Bwd Byts"],
}

def resolve_columns(df: pd.DataFrame) -> pd.DataFrame:
    avail = {c.lower(): c for c in df.columns}
    rename = {}
    for canon, aliases in ALIASES.items():
        for a in aliases:
            if a.lower() in avail:
                rename[avail[a.lower()]] = canon
                break
    out = df.rename(columns=rename).copy()
    for canon in ALIASES.keys():
        if canon not in out.columns:
            out[canon] = 0.0
    return out[list(ALIASES.keys())]

def sanitize_numeric(X: pd.DataFrame) -> pd.DataFrame:
    X = X.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    ql, qh = X.quantile(0.001), X.quantile(0.999)
    X = X.clip(lower=ql, upper=qh, axis=1)
    return X.fillna(0.0)

def main(args):
    df = pd.read_csv(args.csv)
    X = resolve_columns(df)
    X = sanitize_numeric(X.select_dtypes(include="number"))

    km = Pipeline([("base", StandardScaler()), ("km", KMeans(n_clusters=args.k, n_init="auto", random_state=42))])
    iso = Pipeline([("base", StandardScaler()),
                    ("iso", IsolationForest(n_estimators=200, contamination=args.contamination,
                                            random_state=42, n_jobs=-1))])

    km.fit(X)
    iso.fit(X)

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    joblib.dump(km, out / "kmeans.joblib")
    joblib.dump(iso, out / "isoforest.joblib")
    print("Saved:", out / "kmeans.joblib")
    print("Saved:", out / "isoforest.joblib")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out_dir", default="models")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--contamination", type=float, default=0.02)
    main(ap.parse_args())
