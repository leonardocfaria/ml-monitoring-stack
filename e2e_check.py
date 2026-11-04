"""Live check against the compose stack: replay the 3 phases while polling Prometheus; exit 1 unless
FeatureDrift fired during data drift and ConceptDrift fired during concept drift.

docker compose up -d && python e2e_check.py
"""
import sys
import threading
import time

import httpx

from replay import replay

PROM = "http://localhost:9090/api/v1/alerts"


def main(target="http://localhost:8000", per_phase=6000, rps=100) -> int:
    phase, fired, done = ["starting"], set(), threading.Event()

    def poll():
        while not done.wait(5):
            for a in httpx.get(PROM).json()["data"]["alerts"]:
                if a["state"] == "firing":
                    fired.add((phase[0], a["labels"]["alertname"]))

    t = threading.Thread(target=poll)
    t.start()
    try:
        with httpx.Client(base_url=target, timeout=10) as c:
            # replay() logs "phase <name>: ..." when a phase starts: use it to label what the poller sees
            for _ in replay(c, per_phase, rps, log=lambda m: phase.__setitem__(0, m.split()[1].rstrip(":"))):
                pass
        time.sleep(40)  # let the last phase's alert pass its `for:` hold
    finally:
        done.set()
        t.join()
    print("fired:", sorted(fired))
    ok = ("data-drift", "FeatureDrift") in fired and ("concept-drift", "ConceptDrift") in fired \
        and not any(p == "normal" for p, _ in fired)
    print("OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
