"""
separation.py
=============
Dual separation and equivalent martingale measure (EMM) construction via
convex optimisation (cvxpy).

FTAP duality:  no-arbitrage ⟺ existence of Q ~ P such that S is a Q-martingale.

The functions here formalise the dual certificate:
  *  ``solve_separating_density``  — finds a Radon-Nikodym density z with
     z_k > 0 that separates the gains from zero (Hahn–Banach version).
  *  ``solve_martingale_measure``  — directly enforces the martingale
     equalities as linear constraints over terminal probabilities Q_k.
  *  ``verify_martingale_property``/``verify_equivalence`` — sanity checks.

# Using cvxpy for convex feasibility — Context7 confirmed
"""

from __future__ import annotations

import numpy as np
import cvxpy as cp
from typing import Dict, List, Optional, Tuple

from .scenario_tree import ScenarioTree, GainProcess, PredictableStrategy


# ---------------------------------------------------------------------------
# Separating density
# ---------------------------------------------------------------------------

def solve_separating_density(
    tree: ScenarioTree,
    claim_vectors: np.ndarray,
    eps: float = 1e-6,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
    """Find a strictly positive density z separating all claim vectors from 0.

    Solves the feasibility problem:
        z_k > ε                                    (strict positivity / P~Q)
        Σ_k P_k z_k = 1                            (normalisation → Q is prob.)
        Σ_k P_k z_k f_k^i ≤ 0  ∀ i               (separation of each claim)

    where P_k = scenario probability of terminal node k, and f_k^i is the
    k-th terminal payoff of claim i.

    Parameters
    ----------
    tree:          Scenario tree.
    claim_vectors: Terminal payoff matrix, shape (n_claims, n_terminals).
                   Rows are claims; columns index terminal nodes in the order
                   returned by ``tree.get_terminal_nodes()``.
    eps:           Strict-positivity lower bound for z_k.

    Returns
    -------
    z_values : np.ndarray of shape (n_terminals,) or None on infeasibility.
    Q_values : np.ndarray of shape (n_terminals,) — Q_k = P_k * z_k.
    status   : solver status string.
    """
    terminals = tree.get_terminal_nodes()
    K = len(terminals)
    P = np.array([tree.get_scenario_prob(t.node_id) for t in terminals])

    claim_vectors = np.atleast_2d(claim_vectors).astype(float)
    n_claims = claim_vectors.shape[0]

    # Decision variable: z_k  (density at each terminal)
    z = cp.Variable(K, name="z")

    constraints = [
        z >= eps,                              # strict positivity
        P @ z == 1.0,                          # Q is a probability measure
    ]
    for i in range(n_claims):
        # Σ_k P_k z_k f_k^i  =  P ⊙ f^i  @ z  ≤ 0
        constraints.append((P * claim_vectors[i]) @ z <= 0.0)

    # Feasibility problem: minimise 0 subject to constraints
    prob = cp.Problem(cp.Minimize(0), constraints)
    prob.solve(solver=cp.CLARABEL, verbose=False)

    if prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) and z.value is not None:
        z_val = np.array(z.value)
        Q_val = P * z_val
        return z_val, Q_val, prob.status

    return None, None, str(prob.status)


# ---------------------------------------------------------------------------
# Claim matrix construction
# ---------------------------------------------------------------------------

def build_claim_matrix(
    tree: ScenarioTree,
    strategy_list: List[PredictableStrategy],
) -> np.ndarray:
    """Build the terminal payoff matrix for a list of trading strategies.

    Parameters
    ----------
    tree:          Scenario tree.
    strategy_list: List of :class:`~scenario_tree.PredictableStrategy` objects.

    Returns
    -------
    claim_matrix : np.ndarray of shape (n_strategies, n_terminals).
        ``claim_matrix[i, k]`` is the terminal gain (H^i · S)_T on the k-th
        scenario, in the ordering of ``tree.get_terminal_nodes()``.
    """
    terminals = tree.get_terminal_nodes()
    K = len(terminals)
    n_strategies = len(strategy_list)
    claim_matrix = np.zeros((n_strategies, K))

    for i, strategy in enumerate(strategy_list):
        terminal_gains = GainProcess.compute_terminal_gains(tree, strategy)
        for k, terminal in enumerate(terminals):
            claim_matrix[i, k] = terminal_gains.get(terminal.node_id, 0.0)

    return claim_matrix


# ---------------------------------------------------------------------------
# Martingale measure via linear programming
# ---------------------------------------------------------------------------

def solve_martingale_measure(
    tree: ScenarioTree,
    eps: float = 1e-6,
) -> Tuple[Optional[np.ndarray], str]:
    """Solve directly for an equivalent martingale measure (EMM).

    Finds terminal probabilities Q_k > 0 with Σ Q_k = 1 such that S is a
    Q-martingale, i.e. for every internal node n with children c_1, …, c_m:

        Σ_j q_{c_j} · (S_{c_j} − S_n) = 0  (each price dimension)

    where q_{c_j} = Σ_{terminal k ∈ subtree(c_j)} Q_k.

    These linear constraints are expressed directly in terms of {Q_k}.

    Parameters
    ----------
    tree: Scenario tree.
    eps:  Strict-positivity lower bound for Q_k.

    Returns
    -------
    Q_values : np.ndarray of shape (n_terminals,) or None on infeasibility.
    status   : solver status string.
    """
    terminals = tree.get_terminal_nodes()
    K = len(terminals)
    terminal_ids = [t.node_id for t in terminals]
    tid_to_idx = {tid: k for k, tid in enumerate(terminal_ids)}

    # Q: variable over terminal probabilities
    Q = cp.Variable(K, name="Q")

    constraints = [
        Q >= eps,
        cp.sum(Q) == 1.0,
    ]

    # Martingale constraints: for each internal (non-leaf) node
    for node in tree.nodes.values():
        if not node.children_ids:
            continue  # skip terminal nodes

        # Subtree mass for node n: q_n = Σ_{k in subtree(n)} Q_k
        # For each child c_j: q_{c_j} = Σ_{k in subtree(c_j)} Q_k
        # Martingale: Σ_j q_{c_j} * ΔS_{n→c_j} = 0

        children = tree.get_children(node.node_id)
        d = node.state.shape[0]

        for dim in range(d):
            # Build linear combination: Σ_j (Σ_{k in subtree(c_j)} Q_k) * ΔS_j[dim]
            # = Σ_j ΔS_j[dim] * (Σ_{k in subtree(c_j)} Q_k)
            coeff = np.zeros(K)
            for child in children:
                delta_s_dim = child.state[dim] - node.state[dim]
                if abs(delta_s_dim) < 1e-15:
                    continue
                # descendants of child that are terminals
                desc_terminals = tree.get_descendants(child.node_id)
                for desc in desc_terminals:
                    k = tid_to_idx[desc.node_id]
                    coeff[k] += delta_s_dim

            # Constraint: coeff @ Q == 0
            if np.any(np.abs(coeff) > 1e-15):
                constraints.append(coeff @ Q == 0.0)

    prob = cp.Problem(cp.Minimize(0), constraints)
    prob.solve(solver=cp.CLARABEL, verbose=False)

    if prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) and Q.value is not None:
        return np.array(Q.value), prob.status

    return None, str(prob.status)


# ---------------------------------------------------------------------------
# Verification utilities
# ---------------------------------------------------------------------------

def verify_martingale_property(
    tree: ScenarioTree,
    Q_terminal: np.ndarray,
    tol: float = 1e-8,
) -> Dict:
    """Check E_Q[S_t | F_s] = S_s for all internal nodes.

    For each internal node n, computes:
        E_Q[S_{child} | F_n] = Σ_j (q_{c_j} / q_n) · S_{c_j}
    and checks whether this equals S_n within tolerance.

    Parameters
    ----------
    tree:       Scenario tree.
    Q_terminal: EMM probabilities at terminal nodes, shape (n_terminals,).
                Order matches ``tree.get_terminal_nodes()``.
    tol:        Absolute tolerance for martingale residual.

    Returns
    -------
    dict with keys:
        ``is_martingale``      — bool, True if all residuals ≤ tol
        ``max_residual``       — float, largest absolute residual
        ``residuals_by_node``  — dict mapping node_id → per-dimension residual
    """
    terminals = tree.get_terminal_nodes()
    tid_to_idx = {t.node_id: k for k, t in enumerate(terminals)}

    def subtree_mass(node_id: str) -> float:
        """Total Q mass assigned to all terminal descendants of node_id."""
        desc = tree.get_descendants(node_id)
        return sum(Q_terminal[tid_to_idx[d.node_id]] for d in desc)

    residuals_by_node: Dict[str, np.ndarray] = {}
    max_residual = 0.0

    for node in tree.nodes.values():
        if not node.children_ids:
            continue

        q_n = subtree_mass(node.node_id)
        if q_n < 1e-15:
            continue

        # Conditional expected price at immediate children
        children = tree.get_children(node.node_id)
        cond_exp = np.zeros_like(node.state, dtype=float)
        for child in children:
            q_c = subtree_mass(child.node_id)
            cond_exp += (q_c / q_n) * child.state

        residual = np.abs(cond_exp - node.state)
        residuals_by_node[node.node_id] = residual
        max_residual = max(max_residual, float(np.max(residual)))

    return {
        "is_martingale": max_residual <= tol,
        "max_residual": max_residual,
        "residuals_by_node": residuals_by_node,
    }


def verify_equivalence(
    P_terminal: np.ndarray,
    Q_terminal: np.ndarray,
    tol: float = 1e-10,
) -> bool:
    """Check measure equivalence Q ~ P: Q_k > 0 iff P_k > 0.

    Parameters
    ----------
    P_terminal: Physical measure probabilities, shape (K,).
    Q_terminal: Risk-neutral measure probabilities, shape (K,).
    tol:        Threshold below which a probability is considered zero.

    Returns
    -------
    True if Q ~ P (same null sets), False otherwise.
    """
    P_terminal = np.asarray(P_terminal, dtype=float)
    Q_terminal = np.asarray(Q_terminal, dtype=float)

    P_pos = P_terminal > tol
    Q_pos = Q_terminal > tol
    # Equivalence: P_k > 0  ⟺  Q_k > 0
    return bool(np.all(P_pos == Q_pos))
