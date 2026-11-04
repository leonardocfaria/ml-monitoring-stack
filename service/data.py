"""Seeded synthetic stream: 4 features, binary label. shift moves feature means (data drift);
concept=True changes how the label depends on the features (concept drift, inputs unchanged)."""
import numpy as np

W = np.array([1.5, -1.0, 0.8, 0.0])
W_CONCEPT = np.array([-1.5, 1.0, 0.8, 0.0])  # the first two effects flip


def make_data(n: int, seed: int = 0, shift: float = 0.0, concept: bool = False):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, 4))
    X[:, :2] += shift
    w = W_CONCEPT if concept else W
    y = (X @ w + rng.normal(0, 0.5, n) > 0).astype(int)
    return X, y
