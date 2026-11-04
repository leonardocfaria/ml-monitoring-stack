"""Replay a labelled stream against the service: normal -> data drift -> concept drift.
Each prediction's true label is sent to /feedback (delayed ground truth).

python replay.py --target http://localhost:8000 --per-phase 6000 --rps 100
"""
import argparse
import time

import httpx

from service.data import make_data

PHASES = [("normal", dict(shift=0.0)), ("data-drift", dict(shift=1.5)), ("concept-drift", dict(concept=True))]


def replay(client: httpx.Client, per_phase: int = 1500, rps: float = 0, seed: int = 1, phases=PHASES, log=print):
    for k, (name, kw) in enumerate(phases):
        X, y = make_data(per_phase, seed=seed + k, **kw)
        log(f"phase {name}: {per_phase} events")
        for x, label in zip(X, y):
            t = time.perf_counter()
            r = client.post("/predict", json={"features": x.tolist()})
            r.raise_for_status()
            client.post("/feedback", json={"id": r.json()["id"], "label": int(label)}).raise_for_status()
            if rps:
                time.sleep(max(0.0, 1 / rps - (time.perf_counter() - t)))
        yield name


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target", default="http://localhost:8000")
    p.add_argument("--per-phase", type=int, default=6000, help="events per phase; at 100 rps = 60 s, well over the 30 s alert hold")
    p.add_argument("--rps", type=float, default=100, help="0 = as fast as possible")
    a = p.parse_args()
    with httpx.Client(base_url=a.target, timeout=10) as c:
        for _ in replay(c, a.per_phase, a.rps):
            pass
