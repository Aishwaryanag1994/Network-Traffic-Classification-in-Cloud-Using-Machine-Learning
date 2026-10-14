
import os
import time
import logging
from pathlib import Path
from collections import Counter as _Counter
from typing import Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel

from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

from src.config import MODEL_DIR
from src.features.feature_engineering import add_engineered_features

app = FastAPI(title="Traffic Classifier API")


PREDICT_COUNT = Counter("predict_requests_total", "Total predict requests")
ANOMALY_COUNT = Counter("anomaly_requests_total", "Total anomaly requests")

REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["path", "method", "status"],
)
LATENCY = Histogram(
    "request_latency_seconds",
    "Request latency",
    ["path", "method"],
)

PROTOCOL_PREDICT_COUNT = Counter(
    "protocol_predict_requests_total", "Total protocol predict requests"
)
PROTOCOL_CLASS_COUNT = Counter(
    "protocol_predictions_total",
    "Protocol predictions by class",
    ["clazz"],
)
ANOMALIES_FLAGGED_COUNT = Counter(
    "anomalies_flagged_total",
    "Flagged anomalies (count of samples above threshold)",
    ["threshold"],
)
ANOMALY_SCORE = Histogram(
    "anomaly_score",
    "Combined anomaly score distribution",
    buckets=(0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0),
)

QOS_RECO_COUNT = Counter(
    "qos_policy_recommendations_total",
    "QoS policy recommendations made",
    ["policy", "proto_class"],
)

ANOMALY_THRESHOLD = float(os.getenv("ANOMALY_THRESHOLD", "0.5"))


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start = time.perf_counter()
    try:
        response = await call_next(request)
        return response
    finally:
        elapsed = time.perf_counter() - start
        path = request.url.path
        method = request.method
        LATENCY.labels(path=path, method=method).observe(elapsed)
        status = getattr(response, "status_code", 500)
        REQUEST_COUNT.labels(path=path, method=method, status=str(status)).inc()


MODEL_PATH = Path(os.getenv("MODEL_PATH", str(MODEL_DIR / "rf_classifier.joblib")))
KM_PATH = Path(os.getenv("KM_PATH", str(MODEL_DIR / "kmeans.joblib")))
ISO_PATH = Path(os.getenv("ISO_PATH", str(MODEL_DIR / "isoforest.joblib")))
PROTOCOL_MODEL_PATH = Path(
    os.getenv("PROTOCOL_MODEL_PATH", str(MODEL_DIR / "rf_protocol_classifier.joblib"))
)


model = None
km = None
iso = None
protocol_model = None
load_error = None


def _log_versions():
    try:
        import sklearn, numpy, pandas
        logging.info(
            "Versions: sklearn=%s numpy=%s pandas=%s",
            getattr(sklearn, "__version__", "?"),
            getattr(numpy, "__version__", "?"),
            getattr(pandas, "__version__", "?"),
        )
    except Exception:
        logging.exception("Failed to log versions")


@app.on_event("startup")
def load_models_on_startup():
    global model, km, iso, load_error
    _log_versions()
    errs = []

    try:
        if MODEL_PATH.exists():
            model = joblib.load(MODEL_PATH)
            logging.info("Loaded classifier from %s", MODEL_PATH)
        else:
            errs.append(f"Classifier not found at {MODEL_PATH}")
    except Exception as e:
        errs.append(f"classifier: {type(e).__name__}: {e}")
        logging.exception("Failed loading classifier")

    try:
        if KM_PATH.exists():
            km = joblib.load(KM_PATH)
            logging.info("Loaded kmeans pipeline from %s", KM_PATH)
        else:
            errs.append(f"KMeans not found at {KM_PATH}")
    except Exception as e:
        errs.append(f"kmeans: {type(e).__name__}: {e}")
        logging.exception("Failed loading kmeans")

    try:
        if ISO_PATH.exists():
            iso = joblib.load(ISO_PATH)
            logging.info("Loaded IsolationForest pipeline from %s", ISO_PATH)
        else:
            errs.append(f"IsoForest not found at {ISO_PATH}")
    except Exception as e:
        errs.append(f"isoforest: {type(e).__name__}: {e}")
        logging.exception("Failed loading isoforest")

   
    for c in ("http", "dns", "encrypted", "other", "unknown"):
        PROTOCOL_CLASS_COUNT.labels(clazz=c).inc(0)
    ANOMALIES_FLAGGED_COUNT.labels(threshold=str(ANOMALY_THRESHOLD)).inc(0)

    load_error = "; ".join(errs) if errs else None
    if load_error:
        logging.warning("Model load issues: %s", load_error)


class Sample(BaseModel):
    features: Dict[str, float]

class PredictRequest(BaseModel):
    samples: List[Sample]

class PolicyRequest(BaseModel):
    proto_class: str
    anomaly_score: float = 0.0
    malicious_prob: float = 0.0
    latency_ms: Optional[float] = None
    throughput_mbps: Optional[float] = None

class PolicyResponse(BaseModel):
    policy: str
    reason: str
    params: Dict[str, Union[float, int, str]]


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

def _resolve_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map known CICIDS aliases to canonical names and ensure all expected base columns exist."""
    avail = {c.lower(): c for c in df.columns}
    rename = {}
    for canon, aliases in FEATURE_ALIASES.items():
        for alias in aliases:
            if alias.lower() in avail:
                rename[avail[alias.lower()]] = canon
                break
    out = df.copy()
    if rename:
        out = out.rename(columns=rename)
    for canon in FEATURE_ALIASES.keys():
        if canon not in out.columns:
            out[canon] = 0.0
    return out[list(FEATURE_ALIASES.keys())]

def _sanitize_num(X: pd.DataFrame) -> pd.DataFrame:
    X = X.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    ql, qh = X.quantile(0.001), X.quantile(0.999)
    X = X.clip(lower=ql, upper=qh, axis=1)
    return X.fillna(0.0)


@app.get("/healthz")
def healthz():
    return {
        "status": "ok",
        "model_loaded": load_error is None,
        "error": load_error,
        "paths": {
            "rf": str(MODEL_PATH),
            "kmeans": str(KM_PATH),
            "isoforest": str(ISO_PATH),
            "protocol": str(PROTOCOL_MODEL_PATH),
        },
    }

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/model/expected_features")
def expected_features():
    if model is None:
        return {"error": "model not loaded"}
    try:
        cols = model.named_steps["pre"].transformers_[0][2]
        return {"columns": list(cols)}
    except Exception as e:
        return {"error": f"could not extract columns: {e.__class__.__name__}: {e}"}

@app.post("/predict")
def predict(req: PredictRequest):
    PREDICT_COUNT.inc()
    if model is None:
        raise HTTPException(status_code=503, detail=f"Model unavailable: {load_error}")

 
    X_in = pd.DataFrame([s.features for s in req.samples])


    X_base = _resolve_columns(X_in)

    X = add_engineered_features(X_base)

 
    try:
        expected = model.named_steps["pre"].transformers_[0][2]
        X = X.reindex(expected, axis=1, fill_value=0)
    except Exception:
      
        X = X.select_dtypes(include="number").fillna(0)

  
    X = _sanitize_num(X)

    preds = model.predict(X).tolist()
    return {"predictions": preds}

@app.post("/anomaly")
def anomaly(req: PredictRequest):
    ANOMALY_COUNT.inc()
    X_in = pd.DataFrame([s.features for s in req.samples])
    X_base = _resolve_columns(X_in)
    X = add_engineered_features(X_base)
    X = _sanitize_num(X.select_dtypes(include="number"))

    if km is None or iso is None:
        return {"scores": [0.0 for _ in req.samples], "note": "unsupervised models not loaded"}

    try:
        km_expected = km.named_steps["base"].feature_names_in_
        X_km = X.reindex(km_expected, axis=1, fill_value=0)
        baseX_km = km.named_steps["base"].transform(X_km)
        k_labels = km.named_steps["km"].predict(baseX_km)
        centers = km.named_steps["km"].cluster_centers_
        dists = np.sqrt(((baseX_km - centers[k_labels]) ** 2).sum(axis=1))
    except Exception:
        logging.exception("KMeans distance failed")
        dists = [0.0] * len(X)

  
    try:
        iso_expected = iso.named_steps["base"].feature_names_in_
        X_iso = X.reindex(iso_expected, axis=1, fill_value=0)
        baseX_iso = iso.named_steps["base"].transform(X_iso)
        iso_scores = -iso.named_steps["iso"].decision_function(baseX_iso)
    except Exception:
        logging.exception("IsoForest score failed")
        iso_scores = [0.0] * len(X)

    dists = np.asarray(dists, dtype=float).reshape(-1)
    iso_scores = np.asarray(iso_scores, dtype=float).reshape(-1)
    n = len(X)
    if dists.size != n:
        dists = np.zeros(n)
    if iso_scores.size != n:
        iso_scores = np.zeros(n)

    scores = (0.5 * dists) + (0.5 * iso_scores)

    for s in scores:
        ANOMALY_SCORE.observe(float(s))
    flagged = sum(1 for s in scores if float(s) >= ANOMALY_THRESHOLD)
    if flagged:
        ANOMALIES_FLAGGED_COUNT.labels(threshold=str(ANOMALY_THRESHOLD)).inc(flagged)

    return {"scores": [float(s) for s in scores]}



def _bump_protocol_class_counts(preds: List[str]) -> None:
    for clazz, cnt in _Counter(preds).items():
        label = "unknown" if str(clazz).upper() == "OTHER" else str(clazz).lower()
        PROTOCOL_CLASS_COUNT.labels(clazz=label).inc(cnt)

@app.post("/predict_protocol")
def predict_protocol(req: PredictRequest):
    PROTOCOL_PREDICT_COUNT.inc()
    global protocol_model
    if protocol_model is None:
        if PROTOCOL_MODEL_PATH.exists():
            try:
                protocol_model = joblib.load(PROTOCOL_MODEL_PATH)
                logging.info("Loaded protocol model from %s", PROTOCOL_MODEL_PATH)
            except Exception as e:
                logging.exception("Failed to load protocol model")
                raise HTTPException(status_code=500, detail=f"Failed loading protocol model: {e}")
        else:
            raise HTTPException(status_code=503, detail=f"Protocol model not found at {PROTOCOL_MODEL_PATH}")

    X_in = pd.DataFrame([s.features for s in req.samples])
    X_raw = _resolve_columns(X_in)
    X = add_engineered_features(X_raw)
    X = _sanitize_num(X.select_dtypes(include="number"))

    preds = protocol_model.predict(X).tolist()
    _bump_protocol_class_counts(preds)
    return {"predictions": preds}

@app.get("/predict_protocol")
def predict_protocol_get(
    flow_duration: float = None,
    tot_fwd_pkts: float = None,
    tot_bwd_pkts: float = None,
    flow_byts_s: float = None,
    flow_pkts_s: float = None,
    avg_packet_size: float = 0.0,
):
    if any(v is None for v in [flow_duration, tot_fwd_pkts, tot_bwd_pkts, flow_byts_s, flow_pkts_s]):
        return {
            "detail": "Use POST /predict_protocol with JSON body or supply query params, e.g.: "
            "/predict_protocol?flow_duration=1000&tot_fwd_pkts=10&tot_bwd_pkts=3&flow_byts_s=1000&flow_pkts_s=20&avg_packet_size=80"
        }

    global protocol_model
    if protocol_model is None:
        if PROTOCOL_MODEL_PATH.exists():
            try:
                protocol_model = joblib.load(PROTOCOL_MODEL_PATH)
                logging.info("Loaded protocol model from %s", PROTOCOL_MODEL_PATH)
            except Exception as e:
                logging.exception("Failed to load protocol model")
                raise HTTPException(status_code=500, detail=f"Failed loading protocol model: {e}")
        else:
            raise HTTPException(status_code=503, detail=f"Protocol model not found at {PROTOCOL_MODEL_PATH}")

    features = {
        "Flow Duration": flow_duration,
        "Tot Fwd Pkts": tot_fwd_pkts,
        "Tot Bwd Pkts": tot_bwd_pkts,
        "Flow Byts/s": flow_byts_s,
        "Flow Pkts/s": flow_pkts_s,
        "Avg Packet Size": avg_packet_size,
    }
    X_in = pd.DataFrame([features])
    X_raw = _resolve_columns(X_in)
    X = add_engineered_features(X_raw)
    X = _sanitize_num(X.select_dtypes(include="number"))

    pred = protocol_model.predict(X).tolist()[0]
    _bump_protocol_class_counts([pred])
    PROTOCOL_PREDICT_COUNT.inc()
    return {"prediction": pred}


@app.post("/admin/reload_protocol_model")
def reload_protocol_model():
    global protocol_model
    if not PROTOCOL_MODEL_PATH.exists():
        raise HTTPException(status_code=404, detail=f"Model not found at {PROTOCOL_MODEL_PATH}")
    try:
        protocol_model = joblib.load(PROTOCOL_MODEL_PATH)
        return {"status": "ok", "loaded": str(PROTOCOL_MODEL_PATH)}
    except Exception as e:
        logging.exception("Reload failed")
        raise HTTPException(status_code=500, detail=f"Reload failed: {e}")


def _decide_policy(req: PolicyRequest) -> PolicyResponse:
    proto = (req.proto_class or "").upper()
    risk = max(req.anomaly_score or 0.0, req.malicious_prob or 0.0)

    if proto == "VOIP":
        if (req.latency_ms or 0) > 80:
            return PolicyResponse(policy="PRIORITIZE", reason="VOIP with high latency", params={"target_latency_ms": 50})
        return PolicyResponse(policy="PRIORITIZE", reason="VOIP traffic", params={})

    if proto == "P2P":
        if risk >= 0.6 or (req.throughput_mbps or 0) > 20:
            return PolicyResponse(policy="THROTTLE", reason="High-throughput or risky P2P", params={"rate_limit_mbps": 5})

    if proto == "ENCRYPTED" and risk >= 0.7:
        return PolicyResponse(policy="THROTTLE", reason="Encrypted flow with high risk score", params={"rate_limit_mbps": 10})

    if proto in {"HTTP", "DNS"} and risk >= 0.8:
        return PolicyResponse(policy="THROTTLE", reason="High risk HTTP/DNS", params={"rate_limit_mbps": 10})

    return PolicyResponse(policy="NORMAL", reason="No special handling", params={})

@app.post("/qos/policy", response_model=PolicyResponse)
def qos_policy(req: PolicyRequest):
    res = _decide_policy(req)
    QOS_RECO_COUNT.labels(policy=res.policy, proto_class=req.proto_class.upper()).inc()
    return res
