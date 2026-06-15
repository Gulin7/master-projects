import numpy as np
import pandas as pd
from typing import Callable, Tuple


# MU FUNCTIONS (drift)

def mu_constant(mu0: float) -> Callable:
    """Constant drift: μ(t,S) = μ₀"""
    def _mu(t, S):
        return mu0
    _mu.__name__ = f"μ = {mu0} (constant)"
    return _mu


def mu_time_varying(mu0: float, amplitude: float, frequency: float = 1.0) -> Callable:
    """Time-varying drift: μ(t,S) = μ₀ + A·sin(2πft)"""
    def _mu(t, S):
        return mu0 + amplitude * np.sin(2 * np.pi * frequency * t)
    _mu.__name__ = f"μ = {mu0} + {amplitude}·sin(2π·{frequency}·t)"
    return _mu


# SIGMA FUNCTIONS (volatility)

def sigma_constant(sigma0: float) -> Callable:
    """Constant volatility: σ(t,S) = σ₀"""
    def _sigma(t, S):
        return sigma0
    _sigma.__name__ = f"σ = {sigma0} (constant)"
    return _sigma


def sigma_time_varying(sigma0: float, amplitude: float, frequency: float = 1.0) -> Callable:
    """Time-varying volatility: σ(t,S) = σ₀·(1 + A·sin(2πft))"""
    def _sigma(t, S):
        return np.maximum(
            sigma0 * (1 + amplitude * np.sin(2 * np.pi * frequency * t)),
            1e-8
        )
    _sigma.__name__ = f"σ = {sigma0}·(1 + {amplitude}·sin(2π·{frequency}·t))"
    return _sigma


# ALGORITHM 5.1 — Euler-Maruyama simulation
# dS = μ(t,S)·S·dt + σ(t,S)·S·dW

def simulate_black_scholes(
    S0: float,
    T: float,
    n_steps: int,
    n_paths: int,
    mu_func: Callable,
    sigma_func: Callable,
    seed: int | None = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate the Black-Scholes-type SDE:

    dS_t = μ(t,S_t)·S_t·dt + σ(t,S_t)·S_t·dW_t
    """
    if S0 <= 0:
        raise ValueError("S0 must be positive.")
    if T <= 0:
        raise ValueError("T must be positive.")
    if n_steps <= 0:
        raise ValueError("n_steps must be positive.")
    if n_paths <= 0:
        raise ValueError("n_paths must be positive.")

    rng = np.random.default_rng(seed)

    dt = T / n_steps
    time_grid = np.linspace(0, T, n_steps + 1)

    paths = np.zeros((n_paths, n_steps + 1), dtype=float)
    paths[:, 0] = S0

    Z = rng.standard_normal((n_paths, n_steps))

    for i in range(n_steps):
        t_i = time_grid[i]
        S_i = paths[:, i]

        mu_val = mu_func(t_i, S_i)
        sigma_val = sigma_func(t_i, S_i)

        dS = mu_val * S_i * dt + sigma_val * S_i * np.sqrt(dt) * Z[:, i]
        paths[:, i + 1] = np.maximum(S_i + dS, 1e-8)

    return time_grid, paths


# BASIC STATISTICS

def compute_path_statistics(paths: np.ndarray, time_grid: np.ndarray) -> pd.DataFrame:
    """Compute simple statistics of simulated paths over time."""
    return pd.DataFrame({
        "time": time_grid,
        "mean": np.mean(paths, axis=0),
        "std": np.std(paths, axis=0),
        "median": np.median(paths, axis=0),
        "p5": np.percentile(paths, 5, axis=0),
        "p95": np.percentile(paths, 95, axis=0),
    })


