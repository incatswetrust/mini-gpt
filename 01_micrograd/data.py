"""Two interleaving half-circles ("moons"), like sklearn.datasets.make_moons."""
import numpy as np


def make_moons(n_samples=100, noise=0.1, seed=0):
    rng = np.random.default_rng(seed)
    n_top = n_samples // 2
    n_bottom = n_samples - n_top
    t_top = np.linspace(0, np.pi, n_top)
    t_bottom = np.linspace(0, np.pi, n_bottom)
    top = np.stack([np.cos(t_top), np.sin(t_top)], axis=1)
    bottom = np.stack([1 - np.cos(t_bottom), 0.5 - np.sin(t_bottom)], axis=1)
    X = np.concatenate([top, bottom]) + rng.normal(scale=noise, size=(n_samples, 2))
    y = np.concatenate([np.zeros(n_top, dtype=int), np.ones(n_bottom, dtype=int)])
    return X, y
