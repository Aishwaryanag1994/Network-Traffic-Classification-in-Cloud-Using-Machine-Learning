import argparse
from pathlib import Path
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest

from src.config import DATA_DIR, CLASS_LABEL, ID_COLUMNS
from src.data.datasets import load_cicids2017_csvs, prepare_dataset
from src.features.feature_engineering import add_engineered_features

def main(args):
    data_dir = Path(args.data_dir) if args.data_dir else DATA_DIR
    df = load_cicids2017_csvs(data_dir)
    X, y, _ = prepare_dataset(df, CLASS_LABEL, ID_COLUMNS)
    X = add_engineered_features(X)

    base = Pipeline([("imp", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])

    km = Pipeline([("base", base), ("km", KMeans(n_clusters=8, random_state=42, n_init="auto"))])
    km.fit(X)
    iso = Pipeline([("base", base), ("iso", IsolationForest(random_state=42, n_estimators=200, contamination=0.02))])
    iso.fit(X)

    out = Path("models"); out.mkdir(parents=True, exist_ok=True)
    joblib.dump(km, out / "kmeans.joblib", compress=3)
    joblib.dump(iso, out / "isoforest.joblib", compress=3)

    print("Saved unsupervised models to", out)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default=None)
    args = ap.parse_args()
    main(args)
