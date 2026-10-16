"""
local_martingale_example.py
===========================
"exp_7" counterexample: the doubling (Markov coin-toss) process is a
*local* martingale but NOT a true martingale, showing that the class of
local martingales strictly contains the class of true martingales.

Construction
------------
Let t_n = T (1 − 2^{−n}).  At each t_n the process S either doubles
(with probability ½) or collapses to zero (with probability ½) and is then
absorbed.

    S_0 = 1,   S_{t_{n+1}} = 2 S_{t_n} (prob ½),   0 (prob ½, absorbing).

The stopped process S^{σ_m}, where σ_m = inf{t : S_t > 2^m}, IS a true
martingale for every fixed m (bounded stopping theorem), but the full process
satisfies S_T = 0 P-a.s. as N → ∞, so E[S_T] = 0 ≠ S_0 = 1.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Path generation
# ---------------------------------------------------------------------------

def generate_doubling_process_paths(
    T: float,
    N: int,
    n_paths: int,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate Monte Carlo paths of the doubling (Markov coin-toss) process.

    The grid is t_n = T (1 − 2^{−n}) for n = 0, 1, …, N, so t_0 = 0 and
    the grid accumulates toward T from the left.

    Update rule at each step n → n+1:
      * coin = 1 (prob ½) : S_{t_{n+1}} = 2 · S_{t_n}
      * coin = 0 (prob ½) : S_{t_{n+1}} = 0   (absorbing thereafter)

    Parameters
    ----------
    T:       Terminal time.
    N:       Truncation depth (number of coin-toss steps); produces N+1
             time points and N+1 values per path.
    n_paths: Number of Monte Carlo paths to simulate.
    seed:    Random seed for reproducibility.

    Returns
    -------
    paths       : np.ndarray of shape (n_paths, N+1).
    times       : np.ndarray of shape (N+1,) — the grid t_0, t_1, …, t_N.
    coin_tosses : np.ndarray of shape (n_paths, N) — 0/1 coin at each step.
    """
    rng = np.random.default_rng(seed)

    # Time grid: t_n = T * (1 - 2^{-n})
    n_grid = np.arange(N + 1)
    times = T * (1.0 - 2.0 ** (-n_grid))

    # Coin tosses: Bernoulli(0.5) for each path and each step
    coin_tosses = rng.integers(0, 2, size=(n_paths, N)).astype(float)

    # Build paths
    paths = np.zeros((n_paths, N + 1))
    paths[:, 0] = 1.0  # S_0 = 1

    for n in range(N):
        # Wherever the process is already zero, it stays zero
        alive = paths[:, n] > 0
        # Doubling or collapsing
        paths[:, n + 1] = np.where(
            alive,
            np.where(coin_tosses[:, n] == 1,
                     2.0 * paths[:, n],   # double
                     0.0),                # collapse
            0.0,                          # absorbing state
        )

    return paths, times, coin_tosses


# ---------------------------------------------------------------------------
# Stopping times
# ---------------------------------------------------------------------------

def compute_stopping_times_sigma_m(
    paths: np.ndarray,
    times: np.ndarray,
    m: int,
) -> np.ndarray:
    """Compute σ_m = inf{n : S_{t_n} > 2^m} ∧ N for each path.

    Here the stopping time is expressed as an *index* into ``times``
    (and ``paths``), not as a calendar-time value.

    Parameters
    ----------
    paths: Shape (n_paths, N+1) — simulated process values.
    times: Shape (N+1,) — time grid (unused except for length).
    m:     Threshold exponent; we stop when S_t > 2^m.

    Returns
    -------
    sigma_indices : np.ndarray of shape (n_paths,), integer dtype.
        Entry k is the smallest index n with paths[k, n] > 2^m,
        or N (last index) if the threshold is never exceeded.
    """
    n_paths, N_plus_1 = paths.shape
    N = N_plus_1 - 1
    threshold = 2.0 ** m

    sigma_indices = np.full(n_paths, N, dtype=int)
    for i in range(n_paths):
        exceeded = np.where(paths[i] > threshold)[0]
        if len(exceeded) > 0:
            sigma_indices[i] = int(exceeded[0])

    return sigma_indices


def compute_stopped_process(
    paths: np.ndarray,
    sigma_indices: np.ndarray,
) -> np.ndarray:
    """Compute S^{σ_m}_t — the process stopped at σ_m.

    S^{σ_m}_{t_n} = S_{t_{n ∧ σ_m}} = S_{t_n} if n ≤ σ_m, else S_{t_{σ_m}}.

    Parameters
    ----------
    paths:         Shape (n_paths, N+1).
    sigma_indices: Shape (n_paths,) — stopping index per path.

    Returns
    -------
    stopped : np.ndarray of shape (n_paths, N+1).
    """
    n_paths, N_plus_1 = paths.shape
    stopped = paths.copy()

    for i in range(n_paths):
        sigma = sigma_indices[i]
        # For all steps after sigma, hold the value at sigma
        if sigma < N_plus_1 - 1:
            stopped[i, sigma + 1:] = paths[i, sigma]

    return stopped


# ---------------------------------------------------------------------------
# Martingale verification (Monte Carlo)
# ---------------------------------------------------------------------------

def verify_stopped_martingale(
    stopped_paths: np.ndarray,
    times: np.ndarray,
    tol: float = 0.05,
) -> Dict:
    """Check E[S^{σ_m}_{t_{n+1}} | S^{σ_m}_{t_n}] = S^{σ_m}_{t_n} empirically.

    For each step n, group paths by their current value v = S^{σ_m}_{t_n}
    (the process takes finitely many values: 0, 1, 2, 4, …) and compute
    the sample conditional mean of S^{σ_m}_{t_{n+1}} within each group.
    A true martingale has conditional_mean(v) = v.

    Parameters
    ----------
    stopped_paths: Shape (n_paths, N+1) — stopped process paths.
    times:         Shape (N+1,) — time grid.
    tol:           Monte Carlo tolerance for declaring the martingale property
                   satisfied (absolute deviation of conditional mean from v).

    Returns
    -------
    dict with keys:
        ``conditional_means``  — list of dicts {v: E[S_{n+1}|S_n=v]} per step n
        ``residuals``          — list of max absolute residuals per step
        ``is_martingale``      — bool, all residuals ≤ tol
    """
    N = stopped_paths.shape[1] - 1
    conditional_means_list: List[Dict[float, float]] = []
    residuals: List[float] = []

    for n in range(N):
        S_n = stopped_paths[:, n]
        S_n1 = stopped_paths[:, n + 1]
        unique_vals = np.unique(S_n)
        step_cond: Dict[float, float] = {}
        step_res: List[float] = []

        for v in unique_vals:
            mask = S_n == v
            if mask.sum() == 0:
                continue
            cond_mean = float(np.mean(S_n1[mask]))
            step_cond[float(v)] = cond_mean
            step_res.append(abs(cond_mean - v))

        conditional_means_list.append(step_cond)
        residuals.append(float(max(step_res)) if step_res else 0.0)

    is_martingale = all(r <= tol for r in residuals)

    return {
        "conditional_means": conditional_means_list,
        "residuals": residuals,
        "is_martingale": is_martingale,
    }


# ---------------------------------------------------------------------------
# Terminal behaviour
# ---------------------------------------------------------------------------

def verify_terminal_zero(
    paths: np.ndarray,
    tol: float = 0.01,
) -> Dict:
    """Estimate P(S_T = 0) and check that S_T = 0 P-a.s.

    Theoretically, for large N the terminal value S_{t_N} → 0 with
    probability 1 because the probability of never hitting zero is
    (½)^N → 0.

    Parameters
    ----------
    paths: Shape (n_paths, N+1).
    tol:   A path is counted as "terminal zero" if |S_{t_N}| ≤ tol.

    Returns
    -------
    dict with keys:
        ``frac_terminal_zero``  — float, fraction of paths with S_T ≈ 0
        ``mean_terminal``       — float, empirical E[S_T]
        ``is_zero_a_s``         — bool, fraction_terminal_zero ≥ 1 − tol
    """
    terminal = paths[:, -1]
    n_paths = paths.shape[0]
    frac_zero = float(np.mean(np.abs(terminal) <= tol))
    mean_terminal = float(np.mean(terminal))

    return {
        "frac_terminal_zero": frac_zero,
        "mean_terminal": mean_terminal,
        "is_zero_a_s": frac_zero >= 1.0 - tol,
    }


# ---------------------------------------------------------------------------
# Contradiction: no true martingale measure exists
# ---------------------------------------------------------------------------

def check_true_martingale_impossible(
    S0: float,
    terminal_probs_zero: float,
) -> Dict:
    """Show that S cannot be a true martingale given S_T = 0 P-a.s.

    Suppose Q ~ P and S is a Q-martingale.  Then:
        E_Q[S_T] = S_0  (martingale property at T)
    Since Q ~ P and S_T = 0 P-a.s., we have S_T = 0 Q-a.s., so
        E_Q[S_T] = 0 ≠ S_0  (provided S_0 > 0).
    This is a contradiction, so no equivalent true martingale measure exists.

    Parameters
    ----------
    S0:                    Initial asset price (> 0).
    terminal_probs_zero:   Empirical fraction of paths where S_T = 0.

    Returns
    -------
    dict with keys:
        ``contradiction_found`` — bool
        ``expected_terminal``   — float, E_Q[S_T] implied by the P-law (= 0)
        ``required_terminal``   — float, S_0 (required by Q-martingale property)
        ``explanation``         — str, narrative of the contradiction
    """
    # Under P (and hence under any Q ~ P) S_T = 0 a.s., so E_Q[S_T] = 0
    expected_Q_ST = 0.0 * (1.0 - terminal_probs_zero) + 0.0 * terminal_probs_zero
    # (both terms zero; written explicitly to show the reasoning)
    expected_Q_ST = 0.0

    contradiction = (S0 > 0) and (terminal_probs_zero > 0.9)

    explanation = (
        "The doubling process satisfies S_T = 0 P-almost-surely (since the "
        f"probability of never being absorbed is (1/2)^N → 0).  "
        "Any equivalent measure Q must agree on null sets, so S_T = 0 Q-a.s. "
        f"as well.  Hence E_Q[S_T] = 0 ≠ S_0 = {S0}, contradicting the "
        "martingale requirement E_Q[S_T] = S_0.  "
        "Therefore no equivalent true martingale measure exists; S is only a "
        "local martingale (and NFLVR fails for this process at infinite horizon)."
    )

    return {
        "contradiction_found": contradiction,
        "expected_terminal": float(expected_Q_ST),
        "required_terminal": float(S0),
        "explanation": explanation,
    }
