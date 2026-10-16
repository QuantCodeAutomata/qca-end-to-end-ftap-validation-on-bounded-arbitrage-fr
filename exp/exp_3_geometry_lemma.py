"""
Experiment 3: Proof-Mechanism Validation of the Geometry Lemma and
              Common Pathwise Wealth Bound.

Self-contained script verifying the key objects and inequalities from
the Geometry Lemma on a 2-step binomial price system with a redundant
second asset (d=2, S = (Y, Y)).

Lemma objects computed locally:
  u_J      — direction maximising Σ_z ⟨u,z⟩ over unit ball with ⟨u,z⟩≥0 ∀z∈J
  J0       — {z ∈ J : ⟨u_J, z⟩ = 0}
  β        — min_{v ∈ span(J0), |v|=1} max_{z ∈ J0} (−⟨v,z⟩)
  κ        — max(1, 1/β)
  G_t      — projection of H_t onto span(J_t) (reduced strategy)
  R_t      — ∏_{u≤t} (1 + κ·|ΔS_u|)
  V_t      — Σ_{u≤t} |ΔS_u|

Verified inequalities (Lemma 3):
  0 ≤ W_t ≤ a · R_t          for all t, all scenarios
  Var(H·S)_t ≤ a · (R_t − 1)  for all t, all scenarios
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

# ---------------------------------------------------------------------------
# Geometry primitives
# ---------------------------------------------------------------------------

def compute_u_J(J: np.ndarray) -> np.ndarray:
    """
    Maximise Σ_z ⟨u, z⟩ subject to ⟨u, z⟩ ≥ 0 ∀z ∈ J, |u| ≤ 1.

    J : (m, d) array of increment vectors.
    Returns u_J : (d,) unit-ball vector.
    """
    d = J.shape[1]
    c_obj = J.sum(axis=0)   # direction of total-sum objective

    def neg_obj(u):
        return -float(c_obj @ u)

    def neg_obj_grad(u):
        return -c_obj

    constraints = [
        # ⟨u, z⟩ ≥ 0 for each z in J
        {"type": "ineq", "fun": lambda u, z=z: float(z @ u), "jac": lambda u, z=z: z}
        for z in J
    ] + [
        # |u|² ≤ 1
        {"type": "ineq", "fun": lambda u: 1.0 - float(u @ u),
                          "jac": lambda u: -2.0 * u}
    ]

    u0 = np.zeros(d)
    res = minimize(
        neg_obj, u0, jac=neg_obj_grad,
        constraints=constraints,
        method="SLSQP",
        options={"ftol": 1e-12, "maxiter": 500},
    )
    u_opt = res.x
    # Project back onto unit ball if numerical drift
    norm = np.linalg.norm(u_opt)
    if norm > 1.0 + 1e-8:
        u_opt = u_opt / norm
    return u_opt


def compute_projection_matrix(vectors: np.ndarray) -> np.ndarray:
    """
    SVD-based orthogonal projection matrix onto span(vectors).
    vectors : (m, d)  — rows are spanning vectors.
    Returns P : (d, d).
    """
    if vectors.shape[0] == 0:
        d = vectors.shape[1]
        return np.zeros((d, d))
    U, s, Vt = np.linalg.svd(vectors, full_matrices=False)
    tol = 1e-10 * max(s[0], 1.0)
    rank = int(np.sum(s > tol))
    if rank == 0:
        return np.zeros((vectors.shape[1], vectors.shape[1]))
    V = Vt[:rank].T           # (d, rank)
    return V @ V.T            # (d, d) projection


def compute_kappa_J(J: np.ndarray, u_J: np.ndarray, tol: float = 1e-8) -> dict:
    """
    Compute κ_J from the Geometry Lemma.

    J0 = {z ∈ J : ⟨u_J, z⟩ = 0}
    β  = min_{v ∈ span(J0), |v|=1} max_{z ∈ J0} (−⟨v, z⟩)
    κ  = max(1, 1/β)

    Returns dict with J0, beta, kappa, and the projection matrix onto span(J0).
    """
    inner = J @ u_J
    J0_mask = np.abs(inner) < tol * (np.linalg.norm(J, axis=1).max() + 1.0)
    J0 = J[J0_mask]

    if J0.shape[0] == 0:
        return {"J0": J0, "beta": np.inf, "kappa": 1.0, "P_J0": None}

    P_J0 = compute_projection_matrix(J0)
    d = J.shape[1]

    # β = min_{v in span(J0), |v|=1} max_{z in J0} (-<v,z>)
    # Parameterize v over the unit sphere in span(J0) via grid search + refinement.
    # Use orthonormal basis of span(J0):
    U_J0, s_J0, _ = np.linalg.svd(J0, full_matrices=False)
    tol_sv = 1e-10 * max(s_J0[0], 1.0)
    r = int(np.sum(s_J0 > tol_sv))             # rank of J0
    _, _, Vt_J0 = np.linalg.svd(J0, full_matrices=False)
    basis = Vt_J0[:r]                          # (r, d) orthonormal basis of span(J0)

    def max_neg_inner(alpha: np.ndarray) -> float:
        """max_{z in J0} -<v, z>  where v = basis.T @ alpha, |alpha|=1."""
        v = basis.T @ alpha
        return float(np.max(-J0 @ v))

    beta = np.inf
    # Random restarts on the unit sphere in r-dim subspace
    rng = np.random.default_rng(0)
    best_alpha = None
    for _ in range(400):
        alpha0 = rng.standard_normal(r)
        alpha0 /= np.linalg.norm(alpha0)

        # Minimise max_neg_inner (convex in alpha over the sphere)
        def obj(alpha):
            alpha_n = alpha / (np.linalg.norm(alpha) + 1e-14)
            return max_neg_inner(alpha_n)

        cons = [{"type": "eq", "fun": lambda a: np.dot(a, a) - 1.0,
                               "jac": lambda a: 2.0 * a}]
        res = minimize(obj, alpha0, method="SLSQP", constraints=cons,
                       options={"ftol": 1e-12, "maxiter": 300})
        if res.fun < beta:
            beta = res.fun
            best_alpha = res.x

    if beta <= 0:
        kappa = np.inf
    else:
        kappa = max(1.0, 1.0 / beta)

    return {
        "J0":   J0.tolist(),
        "beta": float(beta),
        "kappa": float(kappa),
        "P_J0": P_J0.tolist(),
    }


def project_strategy(H: np.ndarray, J: np.ndarray) -> np.ndarray:
    """
    Reduce H ∈ ℝ^d to G = P_{span(J)} H.
    Verifies ⟨G, z⟩ = ⟨H, z⟩ for all z ∈ J.
    """
    P_J = compute_projection_matrix(J)
    G = P_J @ H
    return G


# ---------------------------------------------------------------------------
# Model: 2-step binomial, d=2, S = (Y, Y) — redundant asset pair
# ---------------------------------------------------------------------------
# Parameters
S0     = np.array([1.0, 1.0])
delta_y = 0.20          # up/down jump size in Y
p_up   = 0.60
p_dn   = 0.40

# Price levels
Y_root = 1.0
Y_up   = 1.2
Y_dn   = 0.8

# Increment vectors at each stage
dS_up = np.array([delta_y,  delta_y])   # (0.2, 0.2)
dS_dn = np.array([-delta_y, -delta_y])  # (-0.2, -0.2)
dS_uu = np.array([0.24,  0.24])
dS_ud = np.array([-0.24, -0.24])
dS_du = np.array([0.16,  0.16])
dS_dd = np.array([-0.16, -0.16])

# 4 paths (uu, ud, du, dd), each with two jumps
PATHS = {
    "uu": {"prob": p_up * p_up, "jumps": [dS_up, dS_uu], "mid": "up"},
    "ud": {"prob": p_up * p_dn, "jumps": [dS_up, dS_ud], "mid": "up"},
    "du": {"prob": p_dn * p_up, "jumps": [dS_dn, dS_du], "mid": "down"},
    "dd": {"prob": p_dn * p_dn, "jumps": [dS_dn, dS_dd], "mid": "down"},
}
PATH_NAMES = ["uu", "ud", "du", "dd"]

# Jump increment sets at each time step (the J in compute_u_J / compute_kappa_J)
J_step1 = np.array([dS_up, dS_dn])       # possible increments at t=0→0.5
J_step2_up = np.array([dS_uu, dS_ud])    # possible increments at t=0.5→1 (mid_up)
J_step2_dn = np.array([dS_du, dS_dd])    # possible increments at t=0.5→1 (mid_down)


# ---------------------------------------------------------------------------
# Wealth process for a given strategy
# ---------------------------------------------------------------------------

def compute_wealth_process(H_root: np.ndarray,
                            H_mid_up: np.ndarray,
                            H_mid_dn: np.ndarray,
                            a: float) -> dict:
    """
    Compute W_t = a + (H·S)_t on all paths.

    H_root  : holding from t=0  to t=0.5
    H_mid_up: holding from t=0.5 to t=1 (given mid_up)
    H_mid_dn: holding from t=0.5 to t=1 (given mid_down)
    a       : initial wealth / admissibility constant

    Returns dict with per-path wealth vectors [W_0, W_0.5, W_1] and variation.
    """
    records = {}
    H_second = {"up": H_mid_up, "down": H_mid_dn}

    for name in PATH_NAMES:
        path = PATHS[name]
        mid   = path["mid"]
        j1, j2 = path["jumps"]

        W0  = a
        X1  = float(H_root @ j1)
        X2  = float(H_second[mid] @ j2)
        W1  = W0 + X1
        W2  = W1 + X2

        records[name] = {
            "prob":       path["prob"],
            "W_0":        W0,
            "W_0.5":      W1,
            "W_1":        W2,
            "gain_jump1": X1,
            "gain_jump2": X2,
            "variation":  abs(X1) + abs(X2),
        }
    return records


# ---------------------------------------------------------------------------
# Compute R_t and V_t processes
# ---------------------------------------------------------------------------

def compute_R_V(kappa_step1: float,
                kappa_step2_up: float,
                kappa_step2_dn: float) -> dict:
    """
    R_t = ∏_{u≤t} (1 + κ_u · |ΔS_u|)   using per-step kappas
    V_t = Σ_{u≤t} |ΔS_u|
    on each path.
    """
    kappa2 = {"up": kappa_step2_up, "down": kappa_step2_dn}
    records = {}
    for name in PATH_NAMES:
        j1, j2 = PATHS[name]["jumps"]
        mid     = PATHS[name]["mid"]
        norm_j1 = float(np.linalg.norm(j1))
        norm_j2 = float(np.linalg.norm(j2))

        R0 = 1.0
        R1 = R0 * (1.0 + kappa_step1   * norm_j1)
        R2 = R1 * (1.0 + kappa2[mid]   * norm_j2)

        V0 = 0.0
        V1 = V0 + norm_j1
        V2 = V1 + norm_j2

        records[name] = {
            "R": [R0, R1, R2],
            "V": [V0, V1, V2],
        }
    return records


# ---------------------------------------------------------------------------
# Verify Lemma 3 inequalities
# ---------------------------------------------------------------------------

def verify_lemma3(wealth_rec: dict, RV_rec: dict, a: float) -> dict:
    """
    Check on every path and every time:
      (a) 0 ≤ W_t ≤ a · R_t
      (b) Var(H·S)_t ≤ a · (R_t − 1)
    """
    times = [0, 0.5, 1.0]
    W_keys = ["W_0", "W_0.5", "W_1"]
    tol = 1e-9

    violations_a = []
    violations_b = []
    residuals = []

    for name in PATH_NAMES:
        wr = wealth_rec[name]
        rv = RV_rec[name]
        var_so_far = 0.0
        jump_gains = [0.0, wr["gain_jump1"], wr["gain_jump2"]]

        for i, (t, wk) in enumerate(zip(times, W_keys)):
            W_t = wr[wk]
            R_t = rv["R"][i]
            V_t = rv["V"][i]
            var_so_far += abs(jump_gains[i])

            ok_a = (-tol <= W_t) and (W_t <= a * R_t + tol)
            ok_b = var_so_far <= a * (R_t - 1.0) + tol

            if not ok_a:
                violations_a.append(
                    f"path={name}, t={t}: W_t={W_t:.6f} NOT in [0, a*R_t={a*R_t:.6f}]"
                )
            if not ok_b:
                violations_b.append(
                    f"path={name}, t={t}: Var={var_so_far:.6f} > a*(R_t-1)={a*(R_t-1):.6f}"
                )
            residuals.append({
                "path": name, "t": t,
                "W_t": round(W_t, 8), "a_R_t": round(a * R_t, 8),
                "R_t": round(R_t, 8), "V_t": round(V_t, 8),
                "variation": round(var_so_far, 8),
                "a_Rm1":    round(a * (R_t - 1.0), 8),
                "ineq_a_ok": ok_a, "ineq_b_ok": ok_b,
            })

    return {
        "violations_a": violations_a,
        "violations_b": violations_b,
        "ineq_a_holds": len(violations_a) == 0,
        "ineq_b_holds": len(violations_b) == 0,
        "residuals":    residuals,
    }


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_figure(all_results: list, kappa: float, save_path: str) -> None:
    """Plot W_t, R_t, V_t for each tested strategy on each path."""
    n_strats = len(all_results)
    fig, axes = plt.subplots(n_strats, 3, figsize=(15, 4 * n_strats), squeeze=False)
    fig.suptitle(f"Exp 3 — Pathwise Wealth, R and V processes  (κ={kappa:.4f})",
                 fontsize=13)

    times = [0.0, 0.5, 1.0]
    colors = {"uu": "royalblue", "ud": "darkorange", "du": "green", "dd": "crimson"}

    for row, rec in enumerate(all_results):
        label = rec["label"]
        wealth_rec = rec["wealth"]
        RV_rec = rec["RV"]
        a = rec["a"]

        ax_W, ax_R, ax_V = axes[row]

        for name in PATH_NAMES:
            wr = wealth_rec[name]
            rv = RV_rec[name]
            W = [wr["W_0"], wr["W_0.5"], wr["W_1"]]
            R = rv["R"]
            V = rv["V"]
            c = colors[name]

            ax_W.plot(times, W, "o-", color=c, label=f"path {name}",
                      lw=1.8, ms=5)
            ax_W.plot(times, [a * r for r in R], "--", color=c, lw=1, alpha=0.5)
            ax_R.plot(times, R, "s-", color=c, lw=1.8, ms=5)
            ax_V.plot(times, V, "^-", color=c, lw=1.8, ms=5)

        ax_W.set_title(f"Strategy {label}: W_t (solid) vs a·R_t (dashed)")
        ax_W.set_ylabel("Wealth W_t")
        ax_W.set_xticks(times)
        if row == 0:
            ax_W.legend(fontsize=7, loc="upper left")

        ax_R.set_title("R_t process")
        ax_R.set_ylabel("R_t")
        ax_R.set_xticks(times)

        ax_V.set_title("V_t process")
        ax_V.set_ylabel("V_t = Σ|ΔS|")
        ax_V.set_xticks(times)

        for ax in (ax_W, ax_R, ax_V):
            ax.set_xlabel("t")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=110)
    plt.close(fig)
    print(f"  Figure saved → {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    os.makedirs("/workspace/project/results", exist_ok=True)
    results = {}

    print("=" * 60)
    print("EXP 3: Geometry Lemma — Pathwise Wealth Bounds Validation")
    print("=" * 60)
    print(f"  d=2 redundant model: S=(Y,Y), δy={delta_y}, p_up={p_up}")

    # ---- Geometry at step 1 ----
    print("\n--- Step 1: Geometry objects at each jump step ---")
    for step_name, J_step in [("step1", J_step1),
                               ("step2_up", J_step2_up),
                               ("step2_dn", J_step2_dn)]:
        u_J = compute_u_J(J_step)
        geo = compute_kappa_J(J_step, u_J)
        print(f"\n  {step_name}: J={J_step.tolist()}")
        print(f"    u_J   = {u_J.round(6).tolist()}")
        print(f"    |u_J| = {np.linalg.norm(u_J):.6f}")
        print(f"    J0    = {geo['J0']}")
        print(f"    β     = {geo['beta']:.6f}")
        print(f"    κ     = {geo['kappa']:.6f}")
        results[f"geometry_{step_name}"] = {
            "J": J_step.tolist(),
            "u_J": u_J.round(8).tolist(),
            **{k: v for k, v in geo.items() if k != "P_J0"},
        }

    # Per-step kappas for the R/V computation
    u_J_step1    = compute_u_J(J_step1)
    geo_step1    = compute_kappa_J(J_step1, u_J_step1)
    kappa_s1     = geo_step1["kappa"]

    u_J_step2_up = compute_u_J(J_step2_up)
    geo_step2_up = compute_kappa_J(J_step2_up, u_J_step2_up)
    kappa_s2_up  = geo_step2_up["kappa"]

    u_J_step2_dn = compute_u_J(J_step2_dn)
    geo_step2_dn = compute_kappa_J(J_step2_dn, u_J_step2_dn)
    kappa_s2_dn  = geo_step2_dn["kappa"]

    # Single representative kappa for summary (max over steps)
    kappa = kappa_s1
    print(f"\n  Per-step kappas: κ_step1={kappa_s1:.6f}, "
          f"κ_step2_up={kappa_s2_up:.6f}, κ_step2_dn={kappa_s2_dn:.6f}")

    # ---- Step 2: Projection preserves gains ----
    print("\n--- Step 2: Projection onto span(J) preserves gains ---")
    test_holdings = [
        ("H=(1,0)",    np.array([1.0,  0.0])),
        ("H=(0,1)",    np.array([0.0,  1.0])),
        ("H=(1,1)",    np.array([1.0,  1.0])),
        ("H=(100,-100)", np.array([100.0, -100.0])),
        ("H=(2,1)",    np.array([2.0,   1.0])),
    ]
    projection_checks = []
    for label, H in test_holdings:
        G = project_strategy(H, J_step1)
        gains_H = [float(H @ z) for z in J_step1]
        gains_G = [float(G @ z) for z in J_step1]
        ok = np.allclose(gains_H, gains_G, atol=1e-9)
        print(f"  {label:18s}: G={G.round(4).tolist()}, "
              f"|G|={np.linalg.norm(G):.4f}, "
              f"gains match={ok}")
        projection_checks.append({
            "H": H.tolist(), "G": G.round(8).tolist(),
            "|G|": round(float(np.linalg.norm(G)), 8),
            "gains_match": ok,
        })
    results["projection_checks"] = projection_checks

    # ---- Step 3-7: Test strategies, verify Lemma 3 ----
    strategies = [
        # label, H_root, H_mid_up, H_mid_dn, a
        ("H_root=(1,0), mid=neutral",
         np.array([1.0,  0.0]), np.array([0.5, 0.5]), np.array([0.0, 0.0]), 1.0),
        ("H_root=(0,1), mid=neutral",
         np.array([0.0,  1.0]), np.array([0.5, 0.5]), np.array([0.0, 0.0]), 1.0),
        ("H_root=(2,1), varying mid",
         np.array([2.0,  1.0]), np.array([1.0, 0.5]), np.array([0.5, 0.5]), 2.0),
        ("H_root=(100,-100) [redundant]",
         np.array([100., -100.]), np.array([0., 0.]), np.array([0., 0.]), 1.0),
        ("H_root=(5,5) [max wealth]",
         np.array([5.0,  5.0]), np.array([5.0, 5.0]), np.array([0., 0.]), 2.0),
    ]

    print("\n--- Steps 3-7: Lemma 3 verification ---")
    all_plot_data = []
    strategy_results = []

    for label, H_r, H_mu, H_md, a in strategies:
        print(f"\n  Strategy: {label}  (a={a})")

        wealth_rec = compute_wealth_process(H_r, H_mu, H_md, a)
        RV_rec     = compute_R_V(kappa_s1, kappa_s2_up, kappa_s2_dn)
        lemma_res  = verify_lemma3(wealth_rec, RV_rec, a)

        # Projection bound check: |G_t| <= κ_t * W_{t-}
        H_dict     = {"up": H_mu, "down": H_md}
        kappa2_dict = {"up": kappa_s2_up, "down": kappa_s2_dn}
        proj_bound_ok = True
        for name in PATH_NAMES:
            wr  = wealth_rec[name]
            mid = PATHS[name]["mid"]

            # step 1: G = proj(H_root, J_step1), W_before = W_0 = a
            G1     = project_strategy(H_r, J_step1)
            bound1 = kappa_s1 * wr["W_0"]
            ok1    = float(np.linalg.norm(G1)) <= bound1 + 1e-6

            # step 2: G = proj(H_mid, J_step2), W_before = W_{0.5}
            J2     = J_step2_up if mid == "up" else J_step2_dn
            G2     = project_strategy(H_dict[mid], J2)
            bound2 = kappa2_dict[mid] * max(wr["W_0.5"], 0.0)
            ok2    = float(np.linalg.norm(G2)) <= bound2 + 1e-6

            if not (ok1 and ok2):
                proj_bound_ok = False

        print(f"    ineq (a) W_t in [0,a·R_t]:    {lemma_res['ineq_a_holds']}")
        print(f"    ineq (b) Var ≤ a·(R_t-1):      {lemma_res['ineq_b_holds']}")
        print(f"    |G_t| ≤ κ·W_{{t-}} bound:         {proj_bound_ok}")
        if lemma_res["violations_a"]:
            for v in lemma_res["violations_a"]:
                print(f"    VIOLATION (a): {v}")
        if lemma_res["violations_b"]:
            for v in lemma_res["violations_b"]:
                print(f"    VIOLATION (b): {v}")

        # Wealth table
        print(f"    {'Path':6s} {'W_0':>8s} {'W_0.5':>8s} {'W_1':>8s} "
              f"{'a·R_T':>8s} {'Var':>8s} {'a(R-1)':>8s}")
        for name in PATH_NAMES:
            wr  = wealth_rec[name]
            rv  = RV_rec[name]
            var = wr["variation"]
            aR  = a * rv["R"][2]
            aRm = a * (rv["R"][2] - 1.0)
            print(f"    {name:6s} {wr['W_0']:8.4f} {wr['W_0.5']:8.4f} "
                  f"{wr['W_1']:8.4f} {aR:8.4f} {var:8.4f} {aRm:8.4f}")

        strategy_results.append({
            "label":            label,
            "ineq_a_holds":     lemma_res["ineq_a_holds"],
            "ineq_b_holds":     lemma_res["ineq_b_holds"],
            "proj_bound_ok":    proj_bound_ok,
            "residuals":        lemma_res["residuals"],
        })
        all_plot_data.append({
            "label":   label,
            "wealth":  wealth_rec,
            "RV":      RV_rec,
            "a":       a,
        })

    results["kappa_step1"]    = kappa_s1
    results["kappa_step2_up"] = kappa_s2_up
    results["kappa_step2_dn"] = kappa_s2_dn
    results["strategy_results"] = strategy_results

    # ---- Summary ----
    all_a_ok = all(r["ineq_a_holds"] for r in strategy_results)
    all_b_ok = all(r["ineq_b_holds"] for r in strategy_results)
    all_proj = all(r["proj_bound_ok"] for r in strategy_results)
    print("\n--- Summary ---")
    print(f"  κ_step1={kappa_s1:.6f}, κ_step2_up={kappa_s2_up:.6f}, "
          f"κ_step2_dn={kappa_s2_dn:.6f}")
    print(f"  All strategies satisfy 0 ≤ W_t ≤ a·R_t:    {all_a_ok}")
    print(f"  All strategies satisfy Var ≤ a·(R_t−1):     {all_b_ok}")
    print(f"  All strategies satisfy |G| ≤ κ_t·W_{{t-}}:   {all_proj}")
    results["summary"] = {
        "kappa_step1":      kappa_s1,
        "kappa_step2_up":   kappa_s2_up,
        "kappa_step2_dn":   kappa_s2_dn,
        "all_ineq_a_holds": all_a_ok,
        "all_ineq_b_holds": all_b_ok,
        "all_proj_bound_ok": all_proj,
        "lemma3_verified": all_a_ok and all_b_ok and all_proj,
    }

    # ---- Figure ----
    print("\n--- Saving figure ---")
    make_figure(all_plot_data[:3], kappa_s1,
                "/workspace/project/results/exp_3_pathwise.png")

    # ---- Save JSON ----
    out_path = "/workspace/project/results/exp_3_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved → {out_path}")
    print("\nEXP 3 COMPLETE.")


if __name__ == "__main__":
    main()
