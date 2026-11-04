"""Data drift (PSI over a sliding window vs training reference) and concept drift (rolling accuracy)."""
from collections import deque

import numpy as np


def psi(ref_frac, cur_frac, eps=1e-4) -> float:
    p, q = np.clip(ref_frac, eps, None), np.clip(cur_frac, eps, None)
    return float(np.sum((q - p) * np.log(q / p)))


class DriftMonitor:
    def __init__(self, reference: np.ndarray, window: int = 500, bins: int = 10):
        """reference: training features (n, d). Bin edges = reference deciles per feature."""
        qs = np.quantile(reference, np.linspace(0, 1, bins + 1), axis=0).T  # (d, bins+1)
        self.edges = [np.r_[-np.inf, q[1:-1], np.inf] for q in qs]
        self.ref = [np.histogram(reference[:, j], e)[0] / len(reference) for j, e in enumerate(self.edges)]
        self.window = deque(maxlen=window)

    def add(self, x) -> None:
        self.window.append(np.asarray(x, dtype=float))

    def psi(self) -> list[float]:
        """PSI per feature of the current window vs the reference (0 until the window has 50 rows)."""
        if len(self.window) < 50:
            return [0.0] * len(self.edges)
        W = np.array(self.window)
        return [psi(r, np.histogram(W[:, j], e)[0] / len(W)) for j, (r, e) in enumerate(zip(self.ref, self.edges))]


class RollingAccuracy:
    def __init__(self, window: int = 200):
        self.hits = deque(maxlen=window)

    def add(self, correct: bool) -> None:
        self.hits.append(bool(correct))

    def value(self) -> float:
        return sum(self.hits) / len(self.hits) if self.hits else float("nan")
