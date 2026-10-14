

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

from src.features.feature_engineering import add_engineered_features

TARGET = "ProtoClass"


FEATURE_ALIASES = {
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
    "Fwd IAT Mean": ["Fwd IAT Mean"],
    "Fwd IAT Std": ["Fwd IAT Std"],
    "Bwd IAT Mean": ["Bwd IAT Mean"],
    "Bwd IAT Std": ["Bwd IAT Std"],
    "Subflow Fwd Bytes": ["Subflow Fwd Bytes", "Subflow Fwd Byts"],
    "Subflow Bwd Bytes": ["Subflow Bwd Bytes", "Subflow Bwd Byts"],
}

def resolve_columns(df: pd.DataFrame) -> pd.DataFrame:
 
    available = {c.lower(): c for c in df.columns}
    renamed = {}
    for canonical, aliases in FEATURE_ALIASES.items():
        for alias in aliases:
            if alias.lower() in available:
                renamed[available[alias.lower()]] = canonical
                break
    work = df.copy()
    if renamed:
        work = work.rename(columns=renamed)
    for canonical in FEATURE_ALIASES.keys():
        if canonical not in work.columns:
            work[canonical] = 0.0
    return work[list(FEATURE_ALIASES.keys())]

def sanitize(X: pd.DataFrame) -> pd.DataFrame:
   
    X = X.apply(pd.to_numeric, errors="coerce")

    X = X.replace([np.inf, -np.inf], np.nan)

    q_low = X.quantile(0.001)
    q_hi = X.quantile(0.999)
    X = X.clip(lower=q_low, upper=q_hi, axis=1)
 
    X = X.fillna(0.0)
    return X

def main(args):
    df = pd.read_csv(args.csv)
    if TARGET not in df.columns:
        raise KeyError(f"Column '{TARGET}' not found in {args.csv}. Run relabel_protocols.py first.")

    
    vc = df[TARGET].value_counts()
    rare = vc[vc < args.min_class_count].index.tolist()
    if rare:
        if args.merge_rare:
            df[TARGET] = df[TARGET].where(~df[TARGET].isin(rare), "OTHER")
            print(f"Merged rare classes into OTHER: {rare}")
        else:
            df = df[~df[TARGET].isin(rare)]
            print(f"Dropped rare classes: {rare}")

    print("Class counts after handling rare:\n", df[TARGET].value_counts())


    X_raw = resolve_columns(df)
    X = add_engineered_features(X_raw).select_dtypes(include="number")
    X = sanitize(X)  
    y = df[TARGET].astype(str)

   
    min_count = y.value_counts().min()
    strat = y if min_count >= 2 else None
    if strat is None:
        print("⚠️ Not using stratify due to very small class counts.")

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.30, random_state=42, stratify=strat)

    clf = RandomForestClassifier(
        n_estimators=300, n_jobs=-1, random_state=42, class_weight="balanced_subsample"
    )
    clf.fit(Xtr, ytr)

    ypred = clf.predict(Xte)
    print(classification_report(yte, ypred, digits=4))
    print("Confusion matrix:\n", confusion_matrix(yte, ypred))

    out_dir = Path(args.model_dir); out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "rf_protocol_classifier.joblib"
    joblib.dump(clf, model_path)
    print("✅ Saved model to:", model_path)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Relabeled CSV produced by relabel_protocols.py")
    ap.add_argument("--model_dir", default="models", help="Where to save the model")
    ap.add_argument("--min_class_count", type=int, default=100,
                    help="Minimum samples per class; below this are merged/dropped")
    ap.add_argument("--merge_rare", action="store_true",
                    help="Merge rare classes into OTHER (default action is drop)")
    args = ap.parse_args()
    main(args)
