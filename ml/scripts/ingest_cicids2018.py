import argparse
import pandas as pd
from pathlib import Path

def ingest(raw_dir, out_file, sample=None):
    raw_dir = Path(raw_dir)
    csvs = sorted(raw_dir.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}")

    frames = []
    for p in csvs:
        print(f"Reading {p} ...")
        df = pd.read_csv(p, low_memory=False)
        frames.append(df)

    df_all = pd.concat(frames, axis=0, ignore_index=True)

  
    df_all = df_all.dropna().drop_duplicates()

    if sample:
        df_all = df_all.sample(n=sample, random_state=42)

    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df_all.to_csv(out_file, index=False)

    print(f"Saved processed dataset to {out_file} with shape {df_all.shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", required=True, help="Directory with raw CICIDS2018 CSVs")
    parser.add_argument("--out", required=True, help="Output processed CSV path")
    parser.add_argument("--sample", type=int, default=None, help="Optional sample size")
    args = parser.parse_args()

    ingest(args.raw_dir, args.out, args.sample)
