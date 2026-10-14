
import argparse, time, random, requests, pandas as pd

BASE_FEATURES = [
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Fwd Packet Length Max",
    "Fwd Packet Length Min",
    "Fwd Packet Length Mean",
    "Bwd Packet Length Max",
    "Bwd Packet Length Min",
    "Bwd Packet Length Mean",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Avg Packet Size",
    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward",
    "Subflow Fwd Bytes",
    "Subflow Bwd Bytes",
]


ALIASES = {
    "Total Fwd Packets":            ["Total Fwd Packets", "Tot Fwd Pkts"],
    "Total Backward Packets":       ["Total Backward Packets", "Tot Bwd Pkts"],
    "Total Length of Fwd Packets":  ["Total Length of Fwd Packets", "TotLen Fwd Pkts"],
    "Total Length of Bwd Packets":  ["Total Length of Bwd Packets", "TotLen Bwd Pkts"],
    "Fwd Packet Length Max":        ["Fwd Packet Length Max", "Fwd Pkt Len Max"],
    "Fwd Packet Length Min":        ["Fwd Packet Length Min", "Fwd Pkt Len Min"],
    "Fwd Packet Length Mean":       ["Fwd Packet Length Mean", "Fwd Pkt Len Mean"],
    "Bwd Packet Length Max":        ["Bwd Packet Length Max", "Bwd Pkt Len Max"],
    "Bwd Packet Length Min":        ["Bwd Packet Length Min", "Bwd Pkt Len Min"],
    "Bwd Packet Length Mean":       ["Bwd Packet Length Mean", "Bwd Pkt Len Mean"],
    "Flow Bytes/s":                 ["Flow Bytes/s", "Flow Byts/s"],
    "Flow Packets/s":               ["Flow Packets/s", "Flow Pkts/s"],
    "Avg Packet Size":              ["Avg Packet Size", "Pkt Size Avg"],
    "Init_Win_bytes_forward":       ["Init_Win_bytes_forward", "Init Fwd Win Byts"],
    "Init_Win_bytes_backward":      ["Init_Win_bytes_backward", "Init Bwd Win Byts"],
    "Subflow Fwd Bytes":            ["Subflow Fwd Bytes", "Subflow Fwd Byts"],
    "Subflow Bwd Bytes":            ["Subflow Bwd Bytes", "Subflow Bwd Byts"],
}

WANTED_CLASSES = ["HTTP", "DNS", "ENCRYPTED", "OTHER"]  

def resolve_row(row: pd.Series) -> dict:
    """Build one sample dict with the canonical feature names filled from aliases."""
    sample = {}
    for canon, alist in ALIASES.items():
        val = None
        for a in alist:
            if a in row and pd.notna(row[a]):
                val = row[a]
                break
        if val is None:
            val = 0.0
        sample[canon] = float(val)
    return sample

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/cicids2018_protocol_labeled.csv")
    ap.add_argument("--url", default="http://127.0.0.1:8000/predict_protocol")
    ap.add_argument("--per-class", type=int, default=80)
    ap.add_argument("--sleep", type=float, default=0.1, help="seconds between posts")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
  
    df = df[df["ProtoClass"].isin(WANTED_CLASSES)].sample(frac=1.0, random_state=42)

   
    needed_any = set(sum(ALIASES.values(), []))  # all alias names
    missing = [c for c in list(needed_any) if c not in df.columns]
 

    counts = {c: 0 for c in WANTED_CLASSES}
    total_target = sum([args.per_class for _ in WANTED_CLASSES])

    for _, row in df.iterrows():
        clazz = row["ProtoClass"]
        if counts[clazz] >= args.per_class:
            continue

        sample = resolve_row(row)
        payload = {"samples": [{"features": sample}]}
        try:
            r = requests.post(args.url, json=payload, timeout=3)
            r.raise_for_status()
            counts[clazz] += 1
        except Exception as e:
         
            if random.random() < 0.01:
                print("post failed:", e)
        time.sleep(args.sleep)

        if sum(counts.values()) >= total_target:
            break

    print("sent:", counts)

if __name__ == "__main__":
    main()
