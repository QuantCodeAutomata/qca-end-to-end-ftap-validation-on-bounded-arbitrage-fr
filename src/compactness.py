"""
compactness.py
==============
L²(μ) norm, forward convex averaging, and compactness objects arising in the
"exp_4" part of the FTAP proof.

The weighted measure μ on (Ω × [0,T]) is defined so that the L²(μ)-ball of
strategies is compact under forward convex averaging.  Concretely, the μ-norm
of a predictable process G is

    ‖G‖²_{L²(μ)} = Σ_{scenario ω}  P(ω) · R_T(ω)^{-3}
                    · Σ_{jump k along ω}  |G_k|² · c_k^{-2} · |ΔS_k|

where R_T is the terminal value of the Lemma-3 bound process and c_k are
strategy-dependent scaling constants.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# μ-norm contributions
# ---------------------------------------------------------------------------

def compute_mu_norm_squared(
    G_path: np.ndarray,
    c_path: np.ndarray,
    delta_S_norms: np.ndarray,
    R_T: float,
    scenario_prob: float,
) -> float:
    """Compute the scenario contribution to ‖G‖²_{L²(μ)}.

    For a single path ω with n_jumps intermediate jumps the contribution is:

        P(ω) · R_T(ω)^{-3} · Σ_k |G_k|² · c_k^{-2} · |ΔS_k|

    Parameters
    ----------
    G_path:        Holding vectors at each jump, shape (n_jumps, d).
    c_path:        Scaling constants c_k > 0, shape (n_jumps,).
    delta_S_norms: Jump norms |ΔS_k|, shape (n_jumps,).
    R_T:           Terminal R process value for this scenario.
    scenario_prob: Scenario probability P(ω).

    Returns
    -------
    Scalar μ-norm-squared contribution for this scenario.
    """
    G_path = np.atleast_2d(G_path).astype(float)
    c_path = np.asarray(c_path, dtype=float)
    delta_S_norms = np.asarray(delta_S_norms, dtype=float)

    n_jumps = G_path.shape[0]
    if n_jumps == 0:
        return 0.0

    # |G_k|² for each jump, shape (n_jumps,)
    G_sq = np.sum(G_path ** 2, axis=1)

    # c_k^{-2}: invert scaling constants (guard against zeros)
    c_inv_sq = np.where(c_path > 0, 1.0 / c_path ** 2, 0.0)

    # Σ_k |G_k|² c_k^{-2} |ΔS_k|
    inner_sum = float(np.dot(G_sq * c_inv_sq, delta_S_norms))

    # Overall factor: P(ω) · R_T^{-3}
    R_T_factor = R_T ** (-3) if R_T > 0 else 0.0

    return scenario_prob * R_T_factor * inner_sum


def compute_total_mu_norm_squared(
    tree,               # ScenarioTree
    strategy,           # PredictableStrategy
    c_func,             # Callable[[str], float]: node_id → c value
    R_terminal: Dict[str, float],
) -> float:
    """Sum ‖G‖²_{L²(μ)} contributions over all terminal scenarios.

    Parameters
    ----------
    tree:       Scenario tree.
    strategy:   Predictable strategy providing holdings at each parent node.
    c_func:     Function mapping a parent node_id to its scaling constant c.
    R_terminal: Mapping terminal_node_id → R_T value for that path.

    Returns
    -------
    Total ‖G‖²_{L²(μ)}.
    """
    from .scenario_tree import GainProcess  # local import avoids circularity

    total = 0.0
    for terminal in tree.get_terminal_nodes():
        path = tree.get_node_path(terminal.node_id)
        scenario_prob = tree.get_scenario_prob(terminal.node_id)
        R_T = R_terminal.get(terminal.node_id, 1.0)

        # Collect jumps along the path
        G_path_list: List[np.ndarray] = []
        c_list: List[float] = []
        dS_norm_list: List[float] = []

        for i in range(len(path) - 1):
            parent = path[i]
            child = path[i + 1]
            delta_S = child.state - parent.state
            dS_norm = float(np.linalg.norm(delta_S))
            if dS_norm < 1e-15:
                continue

            try:
                h = strategy.get_holding(parent.node_id)
            except KeyError:
                h = np.zeros_like(child.state)

            G_path_list.append(h)
            c_list.append(float(c_func(parent.node_id)))
            dS_norm_list.append(dS_norm)

        if not G_path_list:
            continue

        G_path = np.stack(G_path_list)
        c_arr = np.array(c_list)
        dS_arr = np.array(dS_norm_list)

        total += compute_mu_norm_squared(G_path, c_arr, dS_arr, R_T, scenario_prob)

    return total


# ---------------------------------------------------------------------------
# Forward convex averaging
# ---------------------------------------------------------------------------

def forward_convex_average(
    strategies: List[np.ndarray],
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Form a convex combination of strategy arrays.

    In the FTAP proof, forward convex averages G^n = Σ_j λ_j H^{n_j} are
    taken from a sequence of strategies to extract a convergent subsequence
    in L²(μ).

    Parameters
    ----------
    strategies: List of N strategy arrays, each of identical shape
                (n_scenarios, n_jumps, d) or any broadcastable array.
    weights:    Non-negative convex weights summing to 1, shape (N,).
                If ``None`` the uniform distribution 1/N is used.

    Returns
    -------
    Convex combination array with the same shape as each element of
    *strategies*.
    """
    if not strategies:
        raise ValueError("strategies list must not be empty")

    N = len(strategies)
    if weights is None:
        weights = np.full(N, 1.0 / N)

    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 1 or len(weights) != N:
        raise ValueError("weights must be a 1-D array of length len(strategies)")
    if abs(weights.sum() - 1.0) > 1e-8:
        raise ValueError("weights must sum to 1")
    if np.any(weights < 0):
        raise ValueError("weights must be non-negative")

    result = np.zeros_like(strategies[0], dtype=float)
    for w, s in zip(weights, strategies):
        result += w * np.asarray(s, dtype=float)

    return result


# ---------------------------------------------------------------------------
# Gain convergence diagnostic
# ---------------------------------------------------------------------------

def compute_gain_sup_residual(
    gain_sequence: List[np.ndarray],
    gain_limit: np.ndarray,
) -> float:
    """Compute ‖G^n·S − H·S‖_{L^∞} for a sequence of gain paths.

    For each index n in *gain_sequence* computes:
        sup_t |gain_sequence[n][t] − gain_limit[t]|

    and returns the maximum over all n (measuring convergence of the sequence
    to the limit gain path).

    Parameters
    ----------
    gain_sequence: List of gain arrays, each of shape (n_jumps,).
    gain_limit:    Limit gain array, shape (n_jumps,).

    Returns
    -------
    Scalar supremum residual.
    """
    gain_limit = np.asarray(gain_limit, dtype=float)
    if not gain_sequence:
        return 0.0

    max_residual = 0.0
    for gain_n in gain_sequence:
        gain_n = np.asarray(gain_n, dtype=float)
        residual = float(np.max(np.abs(gain_n - gain_limit)))
        max_residual = max(max_residual, residual)

    return max_residual
