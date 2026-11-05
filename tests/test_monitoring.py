import json
import math
import re
from pathlib import Path

import numpy as np
import pytest
import yaml
from fastapi.testclient import TestClient

from replay import PHASES, replay
from service.monitor import DriftMonitor, RollingAccuracy

ROOT = Path(__file__).parent.parent
METRICS = {"model_request_latency_seconds", "model_predictions_total", "model_score", "model_feature_psi",
           "model_rolling_accuracy", "model_feedback_total", "ALERTS"}


def test_drift_monitor_detects_shift_only():
    rng = np.random.default_rng(0)
    m = DriftMonitor(rng.normal(size=(5000, 2)))
    for x in rng.normal(size=(499, 2)) + [5, 5]:
        m.add(x)
    assert m.psi() == [0.0, 0.0]  # window not full yet: no estimate, even for wildly shifted rows
    for x in rng.normal(size=(500, 2)):
        m.add(x)  # replaces the 499 shifted rows
    assert max(m.psi()) < 0.1
    for x in rng.normal(size=(500, 2)) + [1.5, 0]:
        m.add(x)  # the window (500) now holds only shifted rows
    p = m.psi()
    assert p[0] > 0.2 and p[1] < 0.1


def test_rolling_accuracy_window():
    a = RollingAccuracy(window=4)
    assert math.isnan(a.value())
    for hit in (1, 1, 1, 1, 0, 0):
        a.add(hit)
    assert a.value() == 0.5


def metric(text, name, **labels):
    sel = ",".join(f'{k}="{v}"' for k, v in labels.items())
    m = re.search(rf"^{name}{{{sel}}} (\S+)$" if sel else rf"^{name} (\S+)$", text, re.M)
    return float(m.group(1))


@pytest.fixture
def client():
    from service.app import app

    with TestClient(app) as c:
        yield c


def test_api_validation_and_metrics(client):
    assert client.post("/predict", json={"features": [1, 2]}).status_code == 422
    r = client.post("/predict", json={"features": [0.1, 0.2, 0.3, 0.4]}).json()
    assert 0 <= r["score"] <= 1
    assert client.post("/feedback", json={"id": r["id"], "label": r["label"]}).status_code == 200
    assert client.post("/feedback", json={"id": r["id"], "label": 1}).status_code == 404  # only once
    text = client.get("/metrics").text
    assert "model_request_latency_seconds_bucket" in text and metric(text, "model_rolling_accuracy") == 1.0


def test_replay_raises_psi_then_drops_accuracy(client):
    seen = {}
    for phase in replay(client, per_phase=600, phases=PHASES, log=lambda *_: None):
        text = client.get("/metrics").text
        seen[phase] = (max(metric(text, "model_feature_psi", feature=f) for f in ("f0", "f1", "f2", "f3")),
                       metric(text, "model_rolling_accuracy"))
    assert seen["normal"][0] < 0.1 and seen["normal"][1] > 0.85
    assert seen["data-drift"][0] > 0.2  # FeatureDrift would fire
    assert seen["concept-drift"][0] < 0.1  # inputs look normal again...
    assert seen["concept-drift"][1] < 0.75  # ...but ConceptDrift fires


def test_alert_rules_and_dashboard_reference_real_metrics():
    rules = yaml.safe_load((ROOT / "prometheus/alerts.yml").read_text())["groups"][0]["rules"]
    assert {r["alert"] for r in rules} == {"FeatureDrift", "ConceptDrift", "HighP95Latency"}
    dash = json.loads((ROOT / "grafana/dashboards/model.json").read_text())
    exprs = [r["expr"] for r in rules] + [t["expr"] for p in dash["panels"] for t in p["targets"]]
    for e in exprs:
        names = {n for n in re.findall(r"\b(model_[a-z_]+|ALERTS)\b", e)}
        base = {re.sub(r"_(bucket|count|sum)$", "", n) for n in names}
        assert names and base <= METRICS, e
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
    assert set(compose["services"]) == {"model", "prometheus", "grafana"}
    prom = yaml.safe_load((ROOT / "prometheus/prometheus.yml").read_text())
    assert prom["scrape_configs"][0]["static_configs"][0]["targets"] == ["model:8000"]
