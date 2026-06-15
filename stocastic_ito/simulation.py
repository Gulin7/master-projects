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

def _validate_inputs(
    S0: float,
    T: float,
    n_steps: int,
    n_paths: int,
) -> None:
    """Validate simulation inputs."""
    if S0 <= 0:
        raise ValueError("S0 must be positive.")
    if T <= 0:
        raise ValueError("T must be positive.")
    if n_steps <= 0:
        raise ValueError("n_steps must be positive.")
    if n_paths <= 0:
        raise ValueError("n_paths must be positive.")


"""
ALGORITHM Euler-Maruyama simulation
dS = μ(t,S)·S·dt + σ(t,S)·S·dW

Parameters:
S0: float- initial stock price
T: float - total simulation time
n_steps: int - time steps
n_paths: int - independent simulated paths
mu_func: Callable - drift function
sigma_func: Callable - volatility function
seed: int | None = None - a seed would allow to replicate results

Return:
time grid + paths matrix
"""

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
    _validate_inputs(S0, T, n_steps, n_paths)

    rng = np.random.default_rng(seed)

    # dt -> size of a small time interval
    dt = T / n_steps
    # create the timepoints aka 0, dt, 2dt, ..., T
    time_grid = np.linspace(0, T, n_steps + 1)

    # create the matrix to store all simulated prices (for each path, for each step) rows are paths and columns are time steps
    # aka path[a][b] => path a at time step b
    paths = np.zeros((n_paths, n_steps + 1), dtype=float)
    paths[:, 0] = S0

    # simulate a standard normal
    Z = rng.standard_normal((n_paths, n_steps))

   # Use the Euler-Maruyama step -> difference from book one:
   # we first simulate a standard normal N(0,1) and multiply it by sqrt(dt)
   # rather than increment directly N(0, phi) 
    for i in range(n_steps):
        t_i = time_grid[i]
        S_i = paths[:, i]

        mu_val = mu_func(t_i, S_i)
        sigma_val = sigma_func(t_i, S_i)

         # Algorithm:
         # mu_val * s_i * dt -> deterministic, we're just multiplying the 
         # mu function with the current price and time step size

         # sigma_val * s_i * sqrt(dt) * z[:,i] -> random shock
         # Z - N(0,1) => sqrt(dt) * z = N(0, dt)
         # N(0,dt) ~ Brownian increment (dt = step size)

         # i think this works as: var(sqrt(dt) * Z) = dt * Var(Z)
         # Z is standard normally distrib
         # the mean is 0 as the brownian motion has no pref dir

        dS = mu_val * S_i * dt + sigma_val * S_i * np.sqrt(dt) * Z[:, i]
        # we're computing ds + s_i, where s_i is the current stock price
        paths[:, i + 1] = np.maximum(S_i + dS, 1e-8)

    return time_grid, paths

def simulate_black_scholes_book(
    S0: float,
    T: float,
    n_steps: int,
    n_paths: int,
    mu_func: Callable,
    sigma_func: Callable,
    seed: int | None = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Black-Scholes exponential update inspired by Algorithm 5.1:

    S_{j+1} = S_j * exp((μ_j - 0.5 σ_j^2) Δt + σ_j sqrt(Δt) Z_j)

    where Z_j ~ N(0,1).

    For constant μ and σ, this matches the exact one-step GBM update.
    For time-varying μ(t,S) and σ(t,S), this uses left-point evaluation at each step.
    """
    _validate_inputs(S0, T, n_steps, n_paths)

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

        paths[:, i + 1] = S_i * np.exp(
            (mu_val - 0.5 * sigma_val**2) * dt
            + sigma_val * np.sqrt(dt) * Z[:, i]
        )

    return time_grid, paths


# Optional backward-compatible alias
simulate_black_scholes_euler = simulate_black_scholes


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
