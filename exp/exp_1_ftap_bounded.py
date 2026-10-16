"""
Experiment 1: End-to-End FTAP Validation on Bounded Arbitrage-Free Synthetic Price Systems.

Self-contained script verifying the Fundamental Theorem of Asset Pricing (FTAP) on a
2-step binomial discounted price tree with one risky asset (d=1).

Tree structure:
    t=0   : root         S=1.00
    t=0.5 : mid_up       S=1.20  (prob 0.6)
            mid_down     S=0.80  (prob 0.4)
    t=1.0 : term_uu      S=1.44  (prob 0.36)
            term_ud      S=0.96  (prob 0.24)
            term_du      S=0.96  (prob 0.24)
            term_dd      S=0.64  (prob 0.16)

Verification pipeline:
  NA  -> LP infeasibility check
  EMM -> cvxpy separating density / risk-neutral measure
  Q-martingale -> conditional expectation equality
"""

import json
import os

import cvxpy as cp
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linprog

# ---------------------------------------------------------------------------
# 1. Tree definition
# ---------------------------------------------------------------------------

NODES = {
    "root":     {"t": 0.0, "S": 1.00, "parent": None,       "prob": 1.00},
    "mid_up":   {"t": 0.5, "S": 1.20, "parent": "root",     "prob": 0.60},
    "mid_down": {"t": 0.5, "S": 0.80, "parent": "root",     "prob": 0.40},
    "term_uu":  {"t": 1.0, "S": 1.44, "parent": "mid_up",   "prob": 0.60},
    "term_ud":  {"t": 1.0, "S": 0.96, "parent": "mid_up",   "prob": 0.40},
    "term_du":  {"t": 1.0, "S": 0.96, "parent": "mid_down", "prob": 0.60},
    "term_dd":  {"t": 1.0, "S": 0.64, "parent": "mid_down", "prob": 0.40},
}

# Path-level probabilities (joint) and price increments on each of the 4 paths
PATHS = {
    "uu": {"prob": 0.36, "nodes": ["root", "mid_up",   "term_uu"]},
    "ud": {"prob": 0.24, "nodes": ["root", "mid_up",   "term_ud"]},
    "du": {"prob": 0.24, "nodes": ["root", "mid_down", "term_du"]},
    "dd": {"prob": 0.16, "nodes": ["root", "mid_down", "term_dd"]},
}

PATH_NAMES = ["uu", "ud", "du", "dd"]
P = np.array([PATHS[k]["prob"] for k in PATH_NAMES])  # shape (4,)

# State space E (unique discounted price levels, bounded by B=1.44)
E = sorted({NODES[n]["S"] for n in NODES})
B = max(E)


# ---------------------------------------------------------------------------
# 2. Gain matrix
# ---------------------------------------------------------------------------
# Strategy basis: 3 independent predictable choices
#   h_root   : holding h ∈ ℝ from t=0  to t=0.5  (applied at all mid nodes)
#   h_mid_up : holding h ∈ ℝ from t=0.5 to t=1.0 given mid_up
#   h_mid_dn : holding h ∈ ℝ from t=0.5 to t=1.0 given mid_down
#
# Gains X_T = Σ h_u · ΔS_u on each terminal path:
#   uu: h_root*(1.2-1.0) + h_mid_up*(1.44-1.2)  =  0.20·h_r + 0.24·h_u
#   ud: h_root*(1.2-1.0) + h_mid_up*(0.96-1.2)  =  0.20·h_r - 0.24·h_u
#   du: h_root*(0.8-1.0) + h_mid_dn*(0.96-0.8)  = -0.20·h_r + 0.16·h_d
#   dd: h_root*(0.8-1.0) + h_mid_dn*(0.64-0.8)  = -0.20·h_r - 0.16·h_d

A = np.array([
    [ 0.20,  0.24,  0.00],   # uu
    [ 0.20, -0.24,  0.00],   # ud
    [-0.20,  0.00,  0.16],   # du
    [-0.20,  0.00, -0.16],   # dd
])  # shape (4, 3) — rows=paths, cols=strategy basis


def compute_gains(h: np.ndarray) -> np.ndarray:
    """Return terminal gains X_T on all 4 paths for strategy h=(h_root, h_mu, h_md)."""
    return A @ h


# ---------------------------------------------------------------------------
# 3. NA check via linear programming
# ---------------------------------------------------------------------------

def check_no_arbitrage() -> dict:
    """
    LP feasibility: does there exist h such that
        A @ h >= 0   (no terminal loss on any path)
        P @ A @ h >= 1   (positive expected gain — normalization)
    If infeasible → No-Arbitrage holds.
    """
    # A_ub @ h <= b_ub  (scipy convention)
    A_ub = np.vstack([-A, -(P @ A).reshape(1, 3)])   # (5, 3)
    b_ub = np.concatenate([np.zeros(4), [-1.0]])      # (5,)
    bounds = [(-np.inf, np.inf)] * 3                  # h_i unconstrained

    res = linprog(
        c=np.zeros(3),
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=bounds,
        method="highs",
    )
    na_holds = (res.status == 2)  # status 2 = infeasible
    return {"na_holds": bool(na_holds), "lp_status": res.message}


# ---------------------------------------------------------------------------
# 4. Separating density / EMM via cvxpy
# ---------------------------------------------------------------------------

def find_emm(eps: float = 1e-6) -> dict:
    """
    Find equivalent martingale measure Q by solving:
        q >= eps        (strict positivity ≡ Q ~ P)
        sum(q) = 1      (probability measure)
        A.T @ q = 0     (martingale condition for each strategy basis vector)

    Returns Z = q / P (Radon-Nikodym derivative dQ/dP).
    """
    q = cp.Variable(4)
    constraints = [
        q >= eps,
        cp.sum(q) == 1,
        A.T @ q == 0,      # martingale: E_Q[X_T] = 0 for all admissible gains
    ]
    prob = cp.Problem(cp.Minimize(0), constraints)
    prob.solve(solver=cp.SCS, eps=1e-8)

    status = prob.status
    if status in ("infeasible", "infeasible_inaccurate"):
        return {"status": status, "q": None, "Z": None}

    q_val = q.value
    Z_val = q_val / P
    return {
        "status": status,
        "q": q_val,          # Q probabilities on (uu, ud, du, dd)
        "Z": Z_val,          # Radon-Nikodym derivative
        "E_P_Z": float(P @ Z_val),
    }


# ---------------------------------------------------------------------------
# 5. Martingale verification under Q
# ---------------------------------------------------------------------------

def verify_martingale(q: np.ndarray) -> dict:
    """
    Check E_Q[S_T | F_{0.5}] = S_{0.5} at both mid-nodes, and
    E_Q[S_{0.5} | F_0] = S_0 at the root.

    Path indices: uu=0, ud=1, du=2, dd=3.
    """
    # Q conditional on mid_up (paths uu, ud):  renormalise
    q_up = q[:2]
    q_up_cond = q_up / q_up.sum()
    S_T_given_up = np.array([1.44, 0.96])
    E_Q_ST_given_up = float(q_up_cond @ S_T_given_up)

    # Q conditional on mid_down (paths du, dd):
    q_dn = q[2:]
    q_dn_cond = q_dn / q_dn.sum()
    S_T_given_dn = np.array([0.96, 0.64])
    E_Q_ST_given_dn = float(q_dn_cond @ S_T_given_dn)

    # E_Q[S_{0.5}] unconditionally:
    S_mid = np.array([1.20, 1.20, 0.80, 0.80])  # S at t=0.5 for each path
    E_Q_S_mid = float(q @ S_mid)

    return {
        "E_Q_ST_given_up":  E_Q_ST_given_up,
        "S_mid_up":         1.20,
        "martingale_up_ok": abs(E_Q_ST_given_up - 1.20) < 1e-6,
        "E_Q_ST_given_dn":  E_Q_ST_given_dn,
        "S_mid_dn":         0.80,
        "martingale_dn_ok": abs(E_Q_ST_given_dn - 0.80) < 1e-6,
        "E_Q_S_mid":        E_Q_S_mid,
        "S_root":           1.00,
        "martingale_t0_ok": abs(E_Q_S_mid - 1.00) < 1e-6,
    }


# ---------------------------------------------------------------------------
# 6. Figure: tree with Q-probabilities
# ---------------------------------------------------------------------------

def draw_tree(q: np.ndarray, save_path: str) -> None:
    """
    Draw the 2-step binomial price tree annotated with P and Q probabilities.
    """
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(-0.5, 4.5)
    ax.axis("off")
    ax.set_title("2-Step Binomial Tree — FTAP Validation\n"
                 "Black: node price S | Blue: path P-prob | Red: path Q-prob",
                 fontsize=12)

    # Node positions (x=time, y=vertical index)
    pos = {
        "root":     (0.0, 2.0),
        "mid_up":   (1.3, 3.2),
        "mid_down": (1.3, 0.8),
        "term_uu":  (2.6, 4.0),
        "term_ud":  (2.6, 2.6),
        "term_du":  (2.6, 1.4),
        "term_dd":  (2.6, 0.0),
    }

    # Draw edges
    edges = [
        ("root", "mid_up"),
        ("root", "mid_down"),
        ("mid_up", "term_uu"),
        ("mid_up", "term_ud"),
        ("mid_down", "term_du"),
        ("mid_down", "term_dd"),
    ]
    edge_p_labels = {
        ("root", "mid_up"):   f"p={0.60:.2f}",
        ("root", "mid_down"): f"p={0.40:.2f}",
        ("mid_up", "term_uu"):   f"p={0.60:.2f}",
        ("mid_up", "term_ud"):   f"p={0.40:.2f}",
        ("mid_down", "term_du"): f"p={0.60:.2f}",
        ("mid_down", "term_dd"): f"p={0.40:.2f}",
    }

    for (u, v) in edges:
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        ax.plot([x0, x1], [y0, y1], "k-", lw=1.2, zorder=1)
        xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
        ax.text(xm - 0.05, ym + 0.1, edge_p_labels[(u, v)],
                fontsize=8, color="navy", ha="center")

    # Draw nodes
    node_prices = {n: NODES[n]["S"] for n in NODES}
    for name, (x, y) in pos.items():
        ax.scatter(x, y, s=600, zorder=3, color="white", edgecolors="black", linewidths=1.5)
        ax.text(x, y, f"S={node_prices[name]:.2f}", ha="center", va="center",
                fontsize=8, fontweight="bold")

    # Annotate terminal nodes with P and Q
    term_info = {
        "term_uu": (P[0], q[0]),
        "term_ud": (P[1], q[1]),
        "term_du": (P[2], q[2]),
        "term_dd": (P[3], q[3]),
    }
    path_labels = {"term_uu": "uu", "term_ud": "ud", "term_du": "du", "term_dd": "dd"}
    for name, (p_k, q_k) in term_info.items():
        x, y = pos[name]
        ax.text(x + 0.55, y + 0.22, f"P={p_k:.3f}", fontsize=8, color="blue")
        ax.text(x + 0.55, y - 0.08, f"Q={q_k:.3f}", fontsize=8, color="red")
        z_k = q_k / p_k
        ax.text(x + 0.55, y - 0.38, f"Z={z_k:.3f}", fontsize=8, color="green")
        ax.text(x + 0.55, y - 0.65, f"path: {path_labels[name]}", fontsize=7, color="gray")

    # Time axis labels
    for t_val, label in [(0.0, "t=0"), (1.3, "t=0.5"), (2.6, "t=1")]:
        ax.text(t_val, -0.35, label, ha="center", fontsize=9, color="black")

    # Legend patches
    patches = [
        mpatches.Patch(color="blue",  label="P (physical)"),
        mpatches.Patch(color="red",   label="Q (risk-neutral)"),
        mpatches.Patch(color="green", label="Z = dQ/dP"),
    ]
    ax.legend(handles=patches, loc="upper left", fontsize=9)

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

    # -- Tree description --
    print("=" * 60)
    print("EXP 1: FTAP Validation — Bounded Arbitrage-Free Price Tree")
    print("=" * 60)
    print(f"  Assets (d=1), Steps=2, T=1")
    print(f"  S0=1.00, up=1.20, down=0.80, p_up=0.60, p_down=0.40")
    print(f"  State space E = {E}  (|E|={len(E)}, bounded by B={B})")
    print(f"  Path probs P = {dict(zip(PATH_NAMES, P))}")

    results["tree"] = {
        "d": 1, "T": 1, "S0": 1.0, "up": 1.2, "down": 0.8,
        "p_up": 0.6, "p_down": 0.4,
        "state_space_E": E, "bound_B": B,
        "path_probs": dict(zip(PATH_NAMES, P.tolist())),
    }

    # -- NA check --
    print("\n--- Step 3: No-Arbitrage (NA) via LP ---")
    na_result = check_no_arbitrage()
    print(f"  NA holds: {na_result['na_holds']}")
    print(f"  LP status: {na_result['lp_status']}")
    results["na"] = na_result

    # -- EMM construction --
    print("\n--- Step 4-5: Separating Density / EMM via cvxpy ---")
    emm = find_emm()
    print(f"  Solver status: {emm['status']}")

    if emm["q"] is not None:
        q = emm["q"]
        Z = emm["Z"]
        print(f"  Q probabilities: {dict(zip(PATH_NAMES, q.round(6)))}")
        print(f"  Z = dQ/dP      : {dict(zip(PATH_NAMES, Z.round(6)))}")
        print(f"  E_P[Z]         : {emm['E_P_Z']:.8f}  (should be 1.0)")
        print(f"  Q ~ P (all Q_k > 0): {bool(np.all(q > 1e-7))}")
        results["emm"] = {
            "status": emm["status"],
            "Q": dict(zip(PATH_NAMES, q.tolist())),
            "Z": dict(zip(PATH_NAMES, Z.tolist())),
            "E_P_Z": emm["E_P_Z"],
            "Q_equivalent_P": bool(np.all(q > 1e-7)),
        }

        # -- Martingale verification --
        print("\n--- Step 6: Martingale Property of S under Q ---")
        mg = verify_martingale(q)
        print(f"  E_Q[S_T | mid_up]   = {mg['E_Q_ST_given_up']:.6f}  "
              f"(S_mid_up={mg['S_mid_up']})  OK={mg['martingale_up_ok']}")
        print(f"  E_Q[S_T | mid_down] = {mg['E_Q_ST_given_dn']:.6f}  "
              f"(S_mid_dn={mg['S_mid_dn']})  OK={mg['martingale_dn_ok']}")
        print(f"  E_Q[S_{{0.5}} | F_0]  = {mg['E_Q_S_mid']:.6f}  "
              f"(S_root={mg['S_root']})     OK={mg['martingale_t0_ok']}")
        results["martingale"] = mg

        # -- Gain separation check (all basis gains have E_Q = 0) --
        print("\n--- Step 7: Gain Separation (E_Q[X_T^j] = 0 for all basis j) ---")
        gains_EQ = A.T @ q
        for j, g_exp in enumerate(gains_EQ):
            basis_names = ["h_root", "h_mid_up", "h_mid_dn"]
            print(f"  E_Q[gain from {basis_names[j]}] = {g_exp:.2e}")
        results["gain_separation"] = {
            "E_Q_gains": gains_EQ.tolist(),
            "all_zero": bool(np.allclose(gains_EQ, 0, atol=1e-6)),
        }

        # -- Summary --
        nflvr = na_result["na_holds"] and (emm["status"] not in ("infeasible",))
        all_mg_ok = all([mg["martingale_up_ok"], mg["martingale_dn_ok"],
                         mg["martingale_t0_ok"]])
        print("\n--- Summary ---")
        print(f"  NA    = {na_result['na_holds']}")
        print(f"  NFLVR = {nflvr}")
        print(f"  EMM exists = True")
        print(f"  S is Q-martingale = {all_mg_ok}")
        results["summary"] = {
            "NA": na_result["na_holds"],
            "NFLVR": nflvr,
            "EMM_exists": True,
            "S_is_Q_martingale": all_mg_ok,
        }

        # -- Figure --
        print("\n--- Step 10: Saving figure ---")
        draw_tree(q, "/workspace/project/results/exp_1_tree.png")

    # -- Save JSON --
    out_path = "/workspace/project/results/exp_1_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved → {out_path}")
    print("\nEXP 1 COMPLETE.")


if __name__ == "__main__":
    main()
