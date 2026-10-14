

import argparse
from pathlib import Path
import pandas as pd
from .protocol_mapping import protocol_class


def main(args):
    data_path = Path(args.data_dir)

  
    if data_path.is_file() and data_path.suffix.lower() == ".csv":
        print(f"Loading single CSV file: {data_path}")
        df = pd.read_csv(data_path)
    elif data_path.is_dir():
        print(f"Loading CSV files from folder: {data_path}")
        from src.data.datasets import load_cicids2017_csvs
        df = load_cicids2017_csvs(data_path)
    else:
        raise FileNotFoundError(f"{data_path} is not a valid CSV file or folder")

   
    col_map = {c.lower(): c for c in df.columns}

   
    proto_col = col_map.get("protocol")
    if not proto_col:
        raise KeyError(f"Could not find 'Protocol' column. Columns seen: {list(df.columns)[:25]}")

    sp_col = col_map.get("source port") or col_map.get("src port")
    dp_col = col_map.get("destination port") or col_map.get("dst port")

    if not sp_col and not dp_col:
        raise KeyError(
            f"Could not find a source/destination port column. "
            f"Looked for 'Source Port'/'Src Port' and 'Destination Port'/'Dst Port'. "
            f"Columns seen: {list(df.columns)[:25]}"
        )

   
    def infer(row):
        try:
            proto = int(row.get(proto_col, -1))
        except Exception:
            proto = -1
        sport = row.get(sp_col) if sp_col else None
        dport = row.get(dp_col) if dp_col else None
        return protocol_class(proto, sport, dport)

    df["ProtoClass"] = df.apply(infer, axis=1)

    if args.drop_other:
        df = df[df["ProtoClass"] != "OTHER"].copy()

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print("Saved relabeled CSV to:", out)
    print("Class distribution:\n", df["ProtoClass"].value_counts(dropna=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="Path to CSV file OR folder with CICIDS data")
    ap.add_argument("--out_csv", required=True, help="Output CSV path")
    ap.add_argument("--drop_other", action="store_true", help="Drop 'OTHER' rows")
    args = ap.parse_args()
    main(args)
