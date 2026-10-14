from pathlib import Path

DATA_DIR = Path("ml/data/raw/CICIDS2018-test")
MODEL_DIR = Path("models")
REPORTS_DIR = Path("reports")

RANDOM_STATE = 42
N_JOBS = -1
TEST_SIZE = 0.30  

CLASS_LABEL = "Label"
ID_COLUMNS = ["Flow ID","Source IP","Source Port","Destination IP","Destination Port","Protocol","Timestamp"]

BASE_FEATURES = [
    "Flow Duration","Total Fwd Packets","Total Backward Packets","Total Length of Fwd Packets",
    "Total Length of Bwd Packets","Fwd Packet Length Max","Fwd Packet Length Min","Fwd Packet Length Mean",
    "Bwd Packet Length Max","Bwd Packet Length Min","Bwd Packet Length Mean","Flow Bytes/s","Flow Packets/s",
    "Down/Up Ratio","Avg Packet Size","Init_Win_bytes_forward","Init_Win_bytes_backward",
    "Fwd IAT Mean","Fwd IAT Std","Bwd IAT Mean","Bwd IAT Std","Subflow Fwd Bytes","Subflow Bwd Bytes"
]

ENGINEERED_FEATURES = ["LenEntropy","IATCoeffVar","ByteRateRatio"]
