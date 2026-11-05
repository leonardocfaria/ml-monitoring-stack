"""Instrumented model service: /predict, /feedback (delayed ground truth), /metrics (Prometheus)."""
import time
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, Field
from sklearn.linear_model import LogisticRegression

from .data import make_data
from .monitor import DriftMonitor, RollingAccuracy

FEATURES = ["f0", "f1", "f2", "f3"]
LATENCY = Histogram("model_request_latency_seconds", "Time to serve /predict",
                    buckets=(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1))
PREDICTIONS = Counter("model_predictions_total", "Predictions served", ["label"])
SCORE = Histogram("model_score", "Predicted probability of class 1", buckets=[i / 10 for i in range(1, 11)])
PSI = Gauge("model_feature_psi", "PSI of the last 500 requests vs training data", ["feature"])
ACCURACY = Gauge("model_rolling_accuracy", "Accuracy over the last 200 labelled predictions")
FEEDBACK = Counter("model_feedback_total", "Ground-truth labels received")
PSI_EVERY = 50  # recompute PSI every N predictions
# predict/feedback are async with no awaits: they run one at a time on the event loop, so the shared
# window, counters and pending dict need no locks (model inference is < 1 ms)
state = {}


class PredictIn(BaseModel):
    features: list[float] = Field(min_length=4, max_length=4)


class FeedbackIn(BaseModel):
    id: str
    label: int = Field(ge=0, le=1)


@asynccontextmanager
async def lifespan(app):
    X, y = make_data(5000, seed=0)
    state.update(model=LogisticRegression().fit(X, y), drift=DriftMonitor(X), acc=RollingAccuracy(),
                 pending=OrderedDict(), n=0)
    ACCURACY.set(float("nan"))
    for f in FEATURES:
        PSI.labels(f).set(0)
    yield
    state.clear()


app = FastAPI(title="monitored-model", lifespan=lifespan)


@app.post("/predict")
async def predict(req: PredictIn):
    t = time.perf_counter()
    score = float(state["model"].predict_proba([req.features])[0, 1])
    label = int(score > 0.5)
    pid = uuid.uuid4().hex
    pending = state["pending"]
    pending[pid] = label
    if len(pending) > 10_000:  # bounded memory: forget the oldest unlabelled predictions
        pending.popitem(last=False)
    drift = state["drift"]
    drift.add(req.features)
    state["n"] += 1
    if state["n"] % PSI_EVERY == 0:
        for f, v in zip(FEATURES, drift.psi()):
            PSI.labels(f).set(v)
    PREDICTIONS.labels(str(label)).inc()
    SCORE.observe(score)
    LATENCY.observe(time.perf_counter() - t)
    return {"id": pid, "score": score, "label": label}


@app.post("/feedback")
async def feedback(req: FeedbackIn):
    pred = state["pending"].pop(req.id, None)
    if pred is None:
        raise HTTPException(404, "unknown or already labelled prediction id")
    state["acc"].add(pred == req.label)
    ACCURACY.set(state["acc"].value())
    FEEDBACK.inc()
    return {"ok": True}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health")
def health():
    return {"status": "ok"}
