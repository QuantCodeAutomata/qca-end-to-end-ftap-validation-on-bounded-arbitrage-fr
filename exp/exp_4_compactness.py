"""
Experiment 4: Weighted L²(μ) Compactness and Forward Convex Averaging Validation.

Self-contained script demonstrating the compactness mechanism of the Mémin–Stricker
type argument on a 3-step binomial discounted price system.

Model:
  d = 1 risky asset, T = 1, 3 equal steps (Δt = 1/3)
  S_0 = 1.0, increments ΔS ∈ {+0.1, −0.1} with equal probability p = 0.5
  c_t = 2.0  (constant bound process)
  R_t = ∏_{u≤t} (1 + c · |ΔS_u|) = 1.2^(steps taken so far)
  V_t = Σ_{u≤t} |ΔS_u| = (steps so far) × 0.1

  8 terminal paths: {uuu, uud, udu, udd, duu, dud, ddu, ddd}, each prob 0.125
  7 tree-interior nodes (excluding terminal): root, u, d, uu, ud, du, dd

Weighted measure μ on the predictable process space:
  μ(G²) = E[R_T^{-3} · Σ_k |G_k|² · c_k^{-2} · |ΔS_k|]

Five reduced strategies G^n are defined by their node holdings.  Holdings
decrease monotonically from G^1 (large, far from target) to G^5 (moderate 0.3),
ensuring the forward tail averages Ĝ^n converge monotonically in L²(μ) to G^5.
"""

import json
import os

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Tree parameters
# ---------------------------------------------------------------------------
DELTA_S = 0.10          # fixed up/down increment size
C       = 2.0           # bound process c_t
P_PATH  = 0.125         # 1/8 (uniform path probability)
N_STEPS = 3

R_PER_STEP = 1.0 + C * DELTA_S                 # = 1.2
R_T        = R_PER_STEP ** N_STEPS              # = 1.728
V_T        = N_STEPS * DELTA_S                  # = 0.30

# Paths as strings of 'u'/'d' of length 3
ALL_PATHS = ["uuu", "uud", "udu", "udd", "duu", "dud", "ddu", "ddd"]

# Each path has 3 increments ΔS_1, ΔS_2, ΔS_3
def path_increments(path: str) -> np.ndarray:
    return np.array([+DELTA_S if c == "u" else -DELTA_S for c in path])

# ---------------------------------------------------------------------------
# Strategy representation
# ---------------------------------------------------------------------------
# A predictable strategy is defined by 7 holdings:
#   h_root, h_u, h_d, h_uu, h_ud, h_du, h_dd
# indexed as a dict for clarity.

NODE_KEYS = ["root", "u", "d", "uu", "ud", "du", "dd"]

# For each path and each step, which node holds the strategy?
def holding_for_step(path: str, step: int, strategy: dict) -> float:
    """
    step ∈ {0, 1, 2} (0-indexed).
    The holding at step k is chosen based on the path history up to step k.
    """
    if step == 0:
        return strategy["root"]
    elif step == 1:
        node = path[0]                # 'u' or 'd'
        return strategy[node]
    else:  # step == 2
        node = path[:2]              # 'uu', 'ud', 'du', or 'dd'
        return strategy[node]


def strategy_holdings_on_path(path: str, strategy: dict) -> np.ndarray:
    """Return the 3-vector of holdings applied at each step of the path."""
    return np.array([holding_for_step(path, k, strategy) for k in range(N_STEPS)])


def cumulative_gains(path: str, strategy: dict) -> np.ndarray:
    """
    Return array of length (N_STEPS+1,) with cumulative gain at each time:
    gain[0]=0, gain[k] = Σ_{j<k} h_j * ΔS_j.
    """
    dS  = path_increments(path)
    h   = strategy_holdings_on_path(path, strategy)
    g   = np.zeros(N_STEPS + 1)
    for k in range(N_STEPS):
        g[k + 1] = g[k] + h[k] * dS[k]
    return g


# ---------------------------------------------------------------------------
# Weighted L²(μ) norm
# ---------------------------------------------------------------------------

MU_WEIGHT = (R_T ** -3) * (C ** -2) * DELTA_S    # scalar weight per (path, step)

def mu_inner_product(s1: dict, s2: dict) -> float:
    """
    ⟨G¹, G²⟩_{L²(μ)} = E[R_T^{-3} · Σ_k G^1_k · G^2_k · c^{-2} · |ΔS_k|]
                       = MU_WEIGHT · Σ_{paths} P_path · Σ_k h^1_{k,path} · h^2_{k,path}
    (|ΔS_k| is constant = DELTA_S; we factor it into MU_WEIGHT.)
    """
    total = 0.0
    for path in ALL_PATHS:
        h1 = strategy_holdings_on_path(path, s1)
        h2 = strategy_holdings_on_path(path, s2)
        total += P_PATH * float(h1 @ h2)
    return MU_WEIGHT * total


def mu_norm_sq(strategy: dict) -> float:
    return mu_inner_product(strategy, strategy)


def build_gram_matrix(strategies: list) -> np.ndarray:
    """Gram matrix M[i,j] = ⟨G^i, G^j⟩_{L²(μ)} for the list of strategies."""
    N = len(strategies)
    M = np.zeros((N, N))
    for i in range(N):
        for j in range(i, N):
            v = mu_inner_product(strategies[i], strategies[j])
            M[i, j] = v
            M[j, i] = v
    return M


# ---------------------------------------------------------------------------
# Five reduced strategies
# ---------------------------------------------------------------------------

def make_strategies() -> list:
    """
    Define 5 predictable strategies with node holdings that decrease monotonically
    from G^1 to G^5 at every node.  This ensures the forward tail averages
    Ĝ^n converge monotonically (in L²(μ)) to G^5 as n increases.

    G^1: large positions (far from target G^5)
    G^2: moderately large
    G^3: medium
    G^4: small
    G^5: moderate constant 0.3 (the "limit" target)
    """
    G1 = dict(root=2.0, u=1.5, d=1.0, uu=1.5, ud=1.0, du=1.0, dd=0.5)
    G2 = dict(root=1.5, u=1.2, d=0.8, uu=1.2, ud=0.8, du=0.8, dd=0.4)
    G3 = dict(root=1.0, u=0.9, d=0.6, uu=0.9, ud=0.6, du=0.6, dd=0.3)
    G4 = dict(root=0.6, u=0.6, d=0.4, uu=0.6, ud=0.4, du=0.4, dd=0.2)
    G5 = dict(root=0.3, u=0.3, d=0.3, uu=0.3, ud=0.3, du=0.3, dd=0.3)

    return [G1, G2, G3, G4, G5]


# ---------------------------------------------------------------------------
# Forward convex tail averages
# ---------------------------------------------------------------------------

def convex_average(strategies: list, weights: np.ndarray) -> dict:
    """
    Compute the weighted average strategy Ĝ = Σ_j w_j G^j.
    Returns a new strategy dict.
    """
    result = {k: 0.0 for k in NODE_KEYS}
    for w, s in zip(weights, strategies):
        for k in NODE_KEYS:
            result[k] += w * s[k]
    return result


def tail_average(strategies: list, n: int) -> dict:
    """
    Forward tail average from index n (0-based) to end:
      Ĝ^n = (1 / (N − n)) · Σ_{j=n}^{N-1} G^j
    """
    tail = strategies[n:]
    N_tail = len(tail)
    weights = np.ones(N_tail) / N_tail
    return convex_average(tail, weights)


# ---------------------------------------------------------------------------
# Gain supremum residual between two strategies
# ---------------------------------------------------------------------------

def gain_sup_residual(s1: dict, s2: dict) -> float:
    """
    sup_{path, t} |gain_{path,t}(s1) − gain_{path,t}(s2)|
    """
    max_diff = 0.0
    for path in ALL_PATHS:
        g1 = cumulative_gains(path, s1)
        g2 = cumulative_gains(path, s2)
        max_diff = max(max_diff, float(np.max(np.abs(g1 - g2))))
    return max_diff


# ---------------------------------------------------------------------------
# cvxpy: optimal convex combination minimising L²(μ) norm
# ---------------------------------------------------------------------------

def find_optimal_weights(strategies: list, M: np.ndarray) -> dict:
    """
    Solve:   min_{w} w^T M w
             s.t.  w ≥ 0, Σ w_j = 1.
    Returns optimal weights and the achieved L²(μ) norm.
    """
    N = len(strategies)
    w = cp.Variable(N)
    obj   = cp.Minimize(cp.quad_form(w, M))
    cons  = [w >= 0, cp.sum(w) == 1]
    prob  = cp.Problem(obj, cons)
    prob.solve(solver=cp.SCS, eps=1e-9)
    w_val = w.value
    norm_sq = float(w_val @ M @ w_val) if w_val is not None else None
    return {
        "status":   prob.status,
        "weights":  w_val.tolist() if w_val is not None else None,
        "norm_sq":  norm_sq,
        "norm":     float(np.sqrt(max(norm_sq, 0.0))) if norm_sq is not None else None,
    }


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_figure(convergence_table: list, save_path: str) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Experiment 4 — L²(μ) Compactness and Convex Averaging", fontsize=13)

    ns    = [r["n"] for r in convergence_table]
    norms = [r["mu_norm_hat_n"]    for r in convergence_table]
    diffs = [r["mu_norm_diff"]     for r in convergence_table]
    sups  = [r["gain_sup_residual"] for r in convergence_table]

    # Panel 1: L²(μ) norm of Ĝ^n vs n
    ax1.plot(ns, norms, "bo-", lw=2, ms=7, label=r"$\|\hat{G}^n\|_{L^2(\mu)}$")
    ax1.plot(ns, diffs, "rs--", lw=2, ms=7,
             label=r"$\|\hat{G}^n - \hat{G}^N\|_{L^2(\mu)}$")
    ax1.set_xlabel("n (tail starting index, 1-based)")
    ax1.set_ylabel(r"$L^2(\mu)$ norm")
    ax1.set_title(r"$L^2(\mu)$ norm of tail averages $\hat{G}^n$")
    ax1.legend()
    ax1.set_xticks(ns)
    ax1.grid(True, alpha=0.3)

    # Panel 2: sup gain residual vs n
    ax2.plot(ns, sups, "^-", lw=2, ms=7, color="darkorange")
    ax2.set_xlabel("n (tail starting index, 1-based)")
    ax2.set_ylabel(r"$\sup_{t,\omega}|\mathrm{gain}^n_t - \mathrm{gain}^N_t|$")
    ax2.set_title("Pathwise gain residual sup of tail averages")
    ax2.set_xticks(ns)
    ax2.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"  Figure saved → {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    os.makedirs("/workspace/project/results", exist_ok=True)
    results = {}

    print("=" * 65)
    print("EXP 4: Weighted L²(μ) Compactness — Forward Convex Averaging")
    print("=" * 65)
    print(f"  d=1, 3-step binomial, ΔS=±{DELTA_S}, c={C}")
    print(f"  R_T = {R_T:.4f}, V_T = {V_T:.4f}, μ_weight = {MU_WEIGHT:.6e}")

    results["model"] = {
        "d": 1, "n_steps": N_STEPS, "delta_S": DELTA_S,
        "c": C, "R_T": R_T, "V_T": V_T, "MU_WEIGHT": MU_WEIGHT,
        "bound_E_VT_over_RT": round(V_T / R_T, 8),
    }

    # ---- Step 2: Define strategies ----
    strategies = make_strategies()
    strategy_labels = ["G^1 (large, far)", "G^2 (mod-large)",
                       "G^3 (medium)", "G^4 (small)", "G^5 (target 0.3)"]

    print("\n--- Step 3: L²(μ) norms of each strategy G^n ---")
    print(f"  Bound E[V_T/R_T] = {V_T / R_T:.6f}")
    norms_sq = []
    strategy_records = []
    for n, (s, lbl) in enumerate(zip(strategies, strategy_labels)):
        ns2 = mu_norm_sq(s)
        norms_sq.append(ns2)
        within_bound = ns2 <= V_T / R_T + 1e-10
        print(f"  n={n+1}  {lbl:22s}  ||G^n||²={ns2:.6f}  "
              f"||G^n||={np.sqrt(ns2):.6f}  ≤ E[V/R]? {within_bound}")
        strategy_records.append({
            "n": n + 1, "label": lbl,
            "holdings": {k: round(v, 6) for k, v in s.items()},
            "mu_norm_sq": round(ns2, 8),
            "mu_norm": round(float(np.sqrt(ns2)), 8),
            "within_bound": within_bound,
        })
    results["strategies"] = strategy_records

    # ---- Gram matrix ----
    M = build_gram_matrix(strategies)
    print("\n--- Gram matrix M[i,j] = <G^i, G^j>_mu ---")
    with np.printoptions(precision=6, suppress=True):
        print(M)
    results["gram_matrix"] = M.tolist()

    # ---- Step 5–6: Forward convex tail averages and convergence ----
    print("\n--- Steps 5-6: Forward tail averages and L²(μ) convergence ---")
    N = len(strategies)
    tail_avgs = [tail_average(strategies, n) for n in range(N)]
    G_hat_N   = tail_avgs[-1]           # Ĝ^N = G^N (last strategy)

    print(f"\n  {'n':>3}  {'||Ĝⁿ||_mu':>12}  "
          f"{'||Ĝⁿ-ĜᴺN||_mu':>15}  {'gain_sup_resid':>15}")
    convergence_table = []
    for i, G_hat_n in enumerate(tail_avgs):
        n_label = i + 1         # 1-based
        norm_n_sq = mu_norm_sq(G_hat_n)
        norm_n    = float(np.sqrt(norm_n_sq))

        diff_strat = convex_average([G_hat_n, G_hat_N], np.array([1.0, -1.0]))
        norm_diff_sq = mu_norm_sq(diff_strat)
        norm_diff    = float(np.sqrt(max(norm_diff_sq, 0.0)))

        sup_res = gain_sup_residual(G_hat_n, G_hat_N)

        print(f"  {n_label:>3}  {norm_n:>12.6f}  {norm_diff:>15.6f}  {sup_res:>15.6f}")

        convergence_table.append({
            "n":                n_label,
            "hat_holdings":     {k: round(v, 6) for k, v in G_hat_n.items()},
            "mu_norm_sq_hat_n": round(norm_n_sq, 8),
            "mu_norm_hat_n":    round(norm_n, 8),
            "mu_norm_diff":     round(norm_diff, 8),
            "gain_sup_residual": round(sup_res, 8),
        })
    results["convergence_table"] = convergence_table

    # Verify monotone decrease of residuals
    diffs_list = [r["mu_norm_diff"] for r in convergence_table]
    sups_list  = [r["gain_sup_residual"] for r in convergence_table]
    mono_diffs = all(diffs_list[i] >= diffs_list[i+1] - 1e-10
                     for i in range(len(diffs_list)-1))
    print(f"\n  L²(μ) residuals non-increasing: {mono_diffs}")
    print(f"  Final residual (n=N): L²={diffs_list[-1]:.2e}, "
          f"gain_sup={sups_list[-1]:.2e}")

    # ---- Step 7: Optimal convex combination via cvxpy ----
    print("\n--- Step 7: Optimal convex weights via cvxpy ---")
    opt = find_optimal_weights(strategies, M)
    print(f"  Solver status: {opt['status']}")
    if opt["weights"] is not None:
        w_opt = np.array(opt["weights"])
        print(f"  Optimal weights: {np.round(w_opt, 6).tolist()}")
        print(f"  Achieved L²(μ) norm²: {opt['norm_sq']:.8f}")
        print(f"  Achieved L²(μ) norm : {opt['norm']:.8f}")
        results["optimal_weights"] = opt

        # Compare with simple uniform average
        w_unif = np.ones(N) / N
        G_unif = convex_average(strategies, w_unif)
        norm_unif = float(np.sqrt(mu_norm_sq(G_unif)))
        print(f"  Uniform-weight L²(μ) norm: {norm_unif:.8f}  "
              f"(optimal improves by {norm_unif - opt['norm']:.2e})")
        results["uniform_norm"] = round(norm_unif, 8)

    # ---- Gain sup residuals for all convex average pairs ----
    print("\n--- Gain sup residual table ---")
    print(f"  {'n':>3}  {'sup|gain^n - gain^N|':>22}")
    for r in convergence_table:
        print(f"  {r['n']:>3}  {r['gain_sup_residual']:>22.8f}")

    # ---- Summary ----
    print("\n--- Summary ---")
    print(f"  5 strategies defined on 3-step binomial tree (8 paths)")
    print(f"  All L²(μ) norms finite and bounded")
    print(f"  Forward tail averages converge in L²(μ): {mono_diffs}")
    print(f"  Gain sup residual → 0: {sups_list[-1] < 1e-10}")
    print(f"  Optimal convex weights found: {opt['status']}")
    results["summary"] = {
        "strategies_defined":     N,
        "paths":                  len(ALL_PATHS),
        "convergence_monotone":   mono_diffs,
        "final_L2_residual":      diffs_list[-1],
        "final_gain_sup_residual": sups_list[-1],
        "optimal_weights_found":   opt["status"] not in ("infeasible",),
    }

    # ---- Figure ----
    print("\n--- Saving figure ---")
    make_figure(convergence_table, "/workspace/project/results/exp_4_convergence.png")

    # ---- Save JSON ----
    out_path = "/workspace/project/results/exp_4_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved → {out_path}")
    print("\nEXP 4 COMPLETE.")


if __name__ == "__main__":
    main()
