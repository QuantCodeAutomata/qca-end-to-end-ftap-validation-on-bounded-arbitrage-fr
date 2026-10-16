"""
geometry_lemma.py
=================
Geometric objects from Lemma 2 and Lemma 3 in the FTAP proof.

Lemma 2 (Rounding / projection lemma)
    Given a finite set J ⊂ ℝ^d of price increments, find a unit vector u_J
    that is non-negatively correlated with every element of J and maximises
    the number of strict inner products.  The lemma then constructs the
    "contact set" J^0 = {z ∈ J : ⟨u_J, z⟩ = 0}, its orthogonal projector
    P_J, and the quantity β_J that governs the inductive step.

Lemma 3 (Wealth bound)
    The R_t and V_t processes bound the wealth and total-variation of H·S
    from above under any admissible strategy.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.linalg import svd
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Lemma 2 objects
# ---------------------------------------------------------------------------

def compute_u_J(J: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    """Find the Lemma-2 direction u_J for the increment set J.

    Solves
        max_{u ∈ ℝ^d}  Σ_z ⟨u, z⟩
        s.t.  ⟨u, z⟩ ≥ 0  ∀ z ∈ J,   |u| ≤ 1.

    The constraint |u| ≤ 1 ensures a bounded, well-posed problem.  The
    solution u_J satisfies ⟨u_J, z⟩ ≥ 0 for all z ∈ J; J^0 is the subset
    of J that is *perpendicular* to u_J.

    Parameters
    ----------
    J:   Array of shape (m, d) — finite set of price increments.
    tol: Numerical tolerance for constraint satisfaction.

    Returns
    -------
    u_J : np.ndarray of shape (d,).
    """
    J = np.atleast_2d(J).astype(float)
    m, d = J.shape

    # Objective: maximise  u^T c  where  c = Σ_z z  (sum of rows of J)
    c = J.sum(axis=0)

    def objective(u: np.ndarray) -> float:
        return -float(u @ c)  # negate for minimisation

    def objective_grad(u: np.ndarray) -> np.ndarray:
        return -c

    # Constraints:  J @ u >= 0  and  |u|^2 <= 1
    constraints = [
        # ⟨u, z_i⟩ ≥ 0  for every row z_i of J
        {"type": "ineq", "fun": lambda u, zi=J[i]: float(zi @ u),
         "jac": lambda u, zi=J[i]: zi}
        for i in range(m)
    ] + [
        # |u|^2 ≤ 1  ⟺  1 − u^T u ≥ 0
        {"type": "ineq", "fun": lambda u: 1.0 - float(u @ u),
         "jac": lambda u: -2.0 * u},
    ]

    best_val = -np.inf
    best_u = np.zeros(d)

    # Multiple restarts to cope with local minima
    rng = np.random.default_rng(0)
    seeds = [np.zeros(d)] + [rng.standard_normal(d) for _ in range(10)]
    for seed in seeds:
        # Normalise seed so it satisfies |u| ≤ 1
        norm = np.linalg.norm(seed)
        x0 = seed / max(norm, 1.0) * 0.5

        res = minimize(
            objective,
            x0,
            jac=objective_grad,
            method="SLSQP",
            constraints=constraints,
            options={"ftol": 1e-12, "maxiter": 1000},
        )
        if res.success or res.status == 0:
            val = -res.fun
            if val > best_val:
                best_val = val
                best_u = res.x.copy()

    # Project onto unit ball to satisfy |u| ≤ 1 numerically
    norm = np.linalg.norm(best_u)
    if norm > 1.0 + tol:
        best_u /= norm

    return best_u


def compute_J0(J: np.ndarray, u_J: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    """Return J^0 = {z ∈ J : ⟨u_J, z⟩ = 0}.

    Parameters
    ----------
    J:   Array of shape (m, d).
    u_J: Direction vector of shape (d,).
    tol: Absolute tolerance for zero inner product.

    Returns
    -------
    J0 : np.ndarray of shape (m0, d) containing the contact rows.
         Returns empty array of shape (0, d) if no contacts exist.
    """
    J = np.atleast_2d(J).astype(float)
    u_J = np.asarray(u_J, dtype=float)
    inner = J @ u_J
    mask = np.abs(inner) <= tol
    return J[mask]


def compute_projection(J0: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    """Compute P_{J^0}, the orthogonal projector onto span(J^0).

    Uses the economy SVD: the column space of J0^T equals span(J0).

    Parameters
    ----------
    J0:  Array of shape (m0, d) — the contact set.
    tol: Singular-value threshold for rank determination.

    Returns
    -------
    P : np.ndarray of shape (d, d) — orthogonal projection matrix.
        Returns the zero matrix if J0 is empty or has zero rank.
    """
    J0 = np.atleast_2d(J0).astype(float)
    d = J0.shape[1]

    if J0.shape[0] == 0:
        return np.zeros((d, d))

    # SVD of J0: columns of V^T whose singular values exceed tol span J0
    _, s, Vt = svd(J0, full_matrices=False)
    significant = s > tol
    if not np.any(significant):
        return np.zeros((d, d))

    # Basis vectors of span(J0) are the rows of Vt with large singular values
    basis = Vt[significant]   # shape (r, d)
    P = basis.T @ basis       # d×d orthogonal projector onto span(J0)
    return P


def compute_beta_J(J0: np.ndarray, tol: float = 1e-8) -> float:
    """Compute β_J = min_{v ∈ L_J, |v|=1} max_{z ∈ J^0} (−⟨v, z⟩).

    L_J = span(J^0).  If span(J^0) = {0} (or J^0 is empty), return 1.0
    by convention (the inductive step collapses).

    Parameters
    ----------
    J0:  Contact set, shape (m0, d).
    tol: Tolerance for rank / zero detection.

    Returns
    -------
    beta_J : float.
    """
    J0 = np.atleast_2d(J0).astype(float)

    if J0.shape[0] == 0:
        return 1.0

    # Orthonormal basis U (r × d) for span(J0)
    _, s, Vt = svd(J0, full_matrices=False)
    significant = s > tol
    if not np.any(significant):
        return 1.0

    U = Vt[significant]   # shape (r, d), rows are orthonormal basis vectors
    r = U.shape[0]

    # Represent J0 in the basis: A_kj = ⟨z_k, u_j⟩ = (J0 @ U^T)[k,j]
    A = J0 @ U.T   # shape (m0, r)

    # Objective in R^r:  max_k (−alpha^T A[k,:])  subject to |alpha| = 1
    # Reformulate as min t s.t. -alpha^T A[k] ≤ t, |alpha|=1
    # Non-convex (sphere constraint); solve via SLSQP with multiple restarts.
    def obj(alpha: np.ndarray) -> float:
        # max_k(-alpha^T A[k,:]) = max_k(-(A@alpha)[k])
        return float(np.max(-A @ alpha))

    constraints = [
        {"type": "eq", "fun": lambda a: float(a @ a) - 1.0,
         "jac": lambda a: 2.0 * a},
    ]

    best_val = np.inf
    rng = np.random.default_rng(1)
    # Start from random points on the unit sphere in R^r
    n_restarts = max(20, 5 * r)
    candidates = []
    for _ in range(n_restarts):
        v = rng.standard_normal(r)
        v /= np.linalg.norm(v)
        candidates.append(v)

    for x0 in candidates:
        res = minimize(
            obj,
            x0,
            method="SLSQP",
            constraints=constraints,
            options={"ftol": 1e-12, "maxiter": 2000},
        )
        if res.fun < best_val:
            best_val = res.fun

    return float(best_val)


def compute_kappa_J(J: np.ndarray, J0: np.ndarray, beta_J: float) -> float:
    """Compute κ_J = max(1, β_J^{-1}) when span(J^0) ≠ {0}; else 1.

    Parameters
    ----------
    J:       Full increment set, shape (m, d).
    J0:      Contact set, shape (m0, d).
    beta_J:  Value returned by :func:`compute_beta_J`.

    Returns
    -------
    kappa_J : float ≥ 1.
    """
    # If J0 has no non-trivial span, no inductive reduction occurs
    if J0.shape[0] == 0:
        return 1.0

    _, s, _ = svd(J0, full_matrices=False)
    if not np.any(s > 1e-8):
        return 1.0

    if abs(beta_J) < 1e-12:
        # β_J ≈ 0: κ_J is formally +∞; return a large but finite sentinel
        return 1e12

    return float(max(1.0, 1.0 / beta_J))


def compute_all_geometry(J: np.ndarray) -> Dict:
    """Compute all Lemma-2 geometric objects for a finite increment set J.

    Parameters
    ----------
    J: np.ndarray of shape (m, d).

    Returns
    -------
    dict with keys:
        ``u_J``    — Lemma-2 direction, shape (d,)
        ``J0``     — contact set, shape (m0, d)
        ``P_J``    — orthogonal projector onto span(J^0), shape (d, d)
        ``beta_J`` — scalar β_J
        ``kappa_J``— scalar κ_J
        ``rank_J0``— rank of J^0 (dimension of span(J^0))
    """
    J = np.atleast_2d(J).astype(float)
    u_J = compute_u_J(J)
    J0 = compute_J0(J, u_J)
    P_J = compute_projection(J0)
    beta_J = compute_beta_J(J0)
    kappa_J = compute_kappa_J(J, J0, beta_J)

    # rank of span(J0)
    if J0.shape[0] > 0:
        _, s, _ = svd(J0, full_matrices=False)
        rank_J0 = int(np.sum(s > 1e-8))
    else:
        rank_J0 = 0

    return {
        "u_J": u_J,
        "J0": J0,
        "P_J": P_J,
        "beta_J": beta_J,
        "kappa_J": kappa_J,
        "rank_J0": rank_J0,
    }


def project_strategy(
    H: np.ndarray,
    J: np.ndarray,
    W_prev: float,
    tol: float = 1e-8,
) -> np.ndarray:
    """Lemma-2 reduction: project H onto the admissibility-compatible subspace.

    Given current holding H, preceding wealth W_prev > 0, and the set J of
    possible price increments, form:
        J_t = {z ∈ J : ⟨H, z⟩ ≥ −W_prev}   (increments that keep wealth ≥ 0)
        G = P_{J_t} H                           (projection onto span(J_t^0) of J_t)

    Parameters
    ----------
    H:       Current holding vector, shape (d,).
    J:       Possible price increments, shape (m, d).
    W_prev:  Wealth before the jump (must be > 0 for admissibility).
    tol:     Tolerance for constraint inclusion.

    Returns
    -------
    G : np.ndarray of shape (d,) — projected holding.
    """
    H = np.asarray(H, dtype=float)
    J = np.atleast_2d(J).astype(float)

    # Keep increments compatible with admissibility (wealth stays ≥ 0)
    mask = J @ H >= -W_prev - tol
    J_t = J[mask]

    if J_t.shape[0] == 0:
        # All increments violate admissibility; fall back to zero holding
        return np.zeros_like(H)

    # Compute projection onto span of J_t's contact set
    u_Jt = compute_u_J(J_t, tol)
    J0_t = compute_J0(J_t, u_Jt, tol)
    P_Jt = compute_projection(J0_t, tol)

    return P_Jt @ H


# ---------------------------------------------------------------------------
# Lemma 3 objects
# ---------------------------------------------------------------------------

def compute_R_process(
    jump_times: List[float],
    c_values: List[float],
    delta_S_norms: List[float],
) -> np.ndarray:
    """Compute the R_t bounding process at each jump time.

    R_t = Π_{0 < u ≤ t} (1 + c_u · |ΔS_u|)

    R is used in Lemma 3 to bound the wealth process W_t ≤ a · R_t and the
    total variation Var(H·S)_t ≤ a · (R_t − 1).

    Parameters
    ----------
    jump_times:    Ordered list of jump times, length n.
    c_values:      Sequence of c_u scalars at each jump, length n.
    delta_S_norms: |ΔS_u| at each jump, length n.

    Returns
    -------
    R : np.ndarray of shape (n,) with R evaluated just after each jump.
    """
    n = len(jump_times)
    R = np.ones(n)
    running = 1.0
    for k in range(n):
        running *= 1.0 + c_values[k] * delta_S_norms[k]
        R[k] = running
    return R


def compute_V_process(delta_S_list: List[np.ndarray]) -> np.ndarray:
    """Compute V_t = Σ_{0 < u ≤ t} |ΔS_u| — the total-variation process.

    Parameters
    ----------
    delta_S_list: List of n increment vectors ΔS_u ∈ ℝ^d.

    Returns
    -------
    V : np.ndarray of shape (n,) — cumulative total variation after each jump.
    """
    norms = np.array([np.linalg.norm(dS) for dS in delta_S_list])
    return np.cumsum(norms)


def verify_lemma3_inequalities(
    W_path: np.ndarray,
    a: float,
    R_path: np.ndarray,
    gain_path: np.ndarray,
    tol: float = 1e-8,
) -> Dict:
    """Verify the Lemma-3 wealth and variation bounds along a single path.

    Checks:
        1.  0 ≤ W_t              (non-negativity of wealth)
        2.  W_t ≤ a · R_t        (upper bound by discounted R process)
        3.  |Var(H·S)_t| ≤ a · (R_t − 1)  (variation bound; approximated
            by |gain_t| here since H·S is a martingale-like process)

    Parameters
    ----------
    W_path:    Wealth values at each jump time, shape (n,).
    a:         Initial wealth.
    R_path:    R_t values at each jump time, shape (n,).
    gain_path: (H·S)_t values at each jump time, shape (n,).
    tol:       Numerical tolerance.

    Returns
    -------
    dict with keys:
        ``non_negative``        — bool, W_t ≥ −tol everywhere
        ``upper_bound``         — bool, W_t ≤ a·R_t + tol everywhere
        ``variation_bound``     — bool, |gain_t| ≤ a·(R_t−1) + tol
        ``max_W``               — float, max wealth along path
        ``max_residual_upper``  — float, max(W_t − a·R_t)
        ``max_residual_var``    — float, max(|gain_t| − a·(R_t−1))
    """
    W_path = np.asarray(W_path, dtype=float)
    R_path = np.asarray(R_path, dtype=float)
    gain_path = np.asarray(gain_path, dtype=float)

    residual_upper = W_path - a * R_path
    residual_var = np.abs(gain_path) - a * (R_path - 1.0)

    return {
        "non_negative": bool(np.all(W_path >= -tol)),
        "upper_bound": bool(np.all(residual_upper <= tol)),
        "variation_bound": bool(np.all(residual_var <= tol)),
        "max_W": float(np.max(W_path)) if len(W_path) > 0 else 0.0,
        "max_residual_upper": float(np.max(residual_upper)) if len(residual_upper) > 0 else 0.0,
        "max_residual_var": float(np.max(residual_var)) if len(residual_var) > 0 else 0.0,
    }
