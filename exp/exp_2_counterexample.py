"""
Experiment 2: Counterexample — Bounded Synthetic Models with Arbitrage.

Self-contained script demonstrating the failure of NA, the infeasibility of a
separating density, and the non-existence of an equivalent martingale measure
in two explicitly constructed models:

  Model A (d=1, one-jump):  S_T ∈ {1.2, 1.1}, both values > S_0 = 1.0.
    Arbitrage: buy h=1 unit at t=0, sell at T=0.5; gain > 0 always.

  Model B (d=2, one-jump):  two assets, both always move up.
    Same effect; explicit portfolio construction.

Additional: robustness test with multiple h values all showing positive gain.
"""

import json
import os

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linprog

# ---------------------------------------------------------------------------
# Model A: d=1, one-jump, both increments positive
# ---------------------------------------------------------------------------
# Scenarios: k=1 (S goes 1.0→1.2, prob 0.6), k=2 (S goes 1.0→1.1, prob 0.4)
# ΔS = (0.2, 0.1)   — both strictly positive

S0_A = 1.0
S_T_A = np.array([1.2, 1.1])
P_A   = np.array([0.6, 0.4])
dS_A  = S_T_A - S0_A          # = [0.2, 0.1]


def arbitrage_gain_A(h: float) -> np.ndarray:
    """Terminal gain X_T = h * ΔS for strategy h (scalar)."""
    return h * dS_A


def check_na_model_A() -> dict:
    """
    LP check: is there h (free) with h*ΔS >= 0 and E_P[h*ΔS] >= 1?
    Since ΔS = [0.2, 0.1] > 0, h=1 trivially satisfies both.
    """
    # A_ub @ [h] <= b_ub
    A_ub = np.array([
        [-dS_A[0]],          # -0.20 * h <= 0  ⟺  0.20h >= 0
        [-dS_A[1]],          # -0.10 * h <= 0  ⟺  0.10h >= 0
        [-(P_A @ dS_A)],     # -E_P[hΔS] <= -1 ⟺  E_P[hΔS] >= 1
    ])
    b_ub = np.array([0.0, 0.0, -1.0])
    res = linprog(
        c=[0.0],
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=[(-np.inf, np.inf)],
        method="highs",
    )
    na_fails = (res.status != 2)   # feasible ⟹ arbitrage found
    return {
        "na_holds":    not na_fails,
        "lp_status":   res.message,
        "arb_h":       float(res.x[0]) if res.x is not None else None,
        "arb_gain":    arbitrage_gain_A(float(res.x[0])).tolist() if res.x is not None else None,
    }


def try_separating_density_A(eps: float = 1e-6) -> dict:
    """
    Attempt to find separating density Z via cvxpy:
      z_k > 0, Σ P_k z_k = 1, Σ P_k z_k (h*ΔS_k) ≤ 0 for all h.
    The last constraint forces (P ⊙ z) · ΔS = 0, i.e. q · ΔS = 0 where q = P ⊙ z.
    But q_k > 0 and ΔS_k > 0 makes q · ΔS > 0. INFEASIBLE.
    """
    z = cp.Variable(2)
    q = cp.multiply(P_A, z)          # q_k = P_k * z_k

    constraints = [
        z >= eps,
        cp.sum(q) == 1,
        # Separation requires q · ΔS ≤ 0 (for h=+1) AND ≥ 0 (for h=-1),
        # so q · ΔS = 0. But ΔS > 0 and q > 0 ⟹ contradiction.
        q @ dS_A == 0,
    ]
    prob = cp.Problem(cp.Minimize(0), constraints)
    prob.solve(solver=cp.SCS, eps=1e-8)
    return {"status": prob.status, "z": z.value}


def try_emm_A(eps: float = 1e-6) -> dict:
    """
    Attempt to find EMM: q_k > 0, Σ q_k = 1, Σ q_k S_T^k = S_0.
    Martingale constraint:  q_1*1.2 + q_2*1.1 = 1.0
    With q_1+q_2=1, q_k>0:  1.1 + 0.1*q_1 = 1.0  ⟹  q_1 = -1 < 0.  INFEASIBLE.
    """
    q = cp.Variable(2)
    constraints = [
        q >= eps,
        cp.sum(q) == 1,
        q @ S_T_A == S0_A,         # martingale: E_Q[S_T] = S_0
    ]
    prob = cp.Problem(cp.Minimize(0), constraints)
    prob.solve(solver=cp.SCS, eps=1e-8)
    return {"status": prob.status, "q": q.value}


# ---------------------------------------------------------------------------
# Model B: d=2, one-jump, both assets always move up
# ---------------------------------------------------------------------------
# Asset 1: 1.0 → (1.3 or 1.2),  Asset 2: 1.0 → (1.15 or 1.05)
# Both ΔS^1 > 0 and ΔS^2 > 0 in every scenario.

S0_B   = np.array([1.0, 1.0])
S_T_B  = np.array([[1.3, 1.15],   # scenario 1
                   [1.2, 1.05]])  # scenario 2
P_B    = np.array([0.5, 0.5])
dS_B   = S_T_B - S0_B             # shape (2 scenarios, 2 assets)


def check_na_model_B() -> dict:
    """
    LP: find portfolio (h1, h2) s.t. dS_B @ h >= 0 and E_P[dS_B @ h] >= 1.
    dS_B = [[0.3, 0.15], [0.2, 0.05]] — all positive.
    h = (1, 0) satisfies this trivially.
    """
    # Variables: h = [h1, h2]
    # Constraints: -dS_B @ h <= 0, -P_B @ dS_B @ h <= -1
    A_ub = np.vstack([-dS_B, -(P_B @ dS_B).reshape(1, 2)])   # (3, 2)
    b_ub = np.array([0.0, 0.0, -1.0])
    res = linprog(
        c=[0.0, 0.0],
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=[(-np.inf, np.inf), (-np.inf, np.inf)],
        method="highs",
    )
    na_fails = (res.status != 2)
    arb_h = res.x.tolist() if res.x is not None else None
    arb_gain = (dS_B @ res.x).tolist() if res.x is not None else None
    return {
        "na_holds":  not na_fails,
        "lp_status": res.message,
        "arb_h":     arb_h,
        "arb_gain":  arb_gain,
    }


def try_emm_B(eps: float = 1e-6) -> dict:
    """
    Attempt EMM for d=2 model:
      q_k > 0, Σ q_k = 1, Σ q_k S_T^k_j = S_0_j for j=1,2.
    Both martingale constraints are violated since all terminal prices > initial.
    """
    q = cp.Variable(2)
    constraints = [
        q >= eps,
        cp.sum(q) == 1,
        S_T_B.T @ q == S0_B,   # E_Q[S_T^j] = S_0^j for each asset j
    ]
    prob = cp.Problem(cp.Minimize(0), constraints)
    prob.solve(solver=cp.SCS, eps=1e-8)
    return {"status": prob.status, "q": q.value}


# ---------------------------------------------------------------------------
# Robustness: multiple h values, all showing arbitrage
# ---------------------------------------------------------------------------

def robustness_test_A() -> list:
    """Test multiple h values confirming NA fails for all h > 0."""
    h_values = [0.5, 1.0, 2.0, 5.0, 10.0]
    records = []
    for h in h_values:
        gain = arbitrage_gain_A(h)
        is_arb = bool(np.all(gain >= 0) and np.any(gain > 0))
        records.append({
            "h":          h,
            "gain":       gain.tolist(),
            "is_arb":     is_arb,
            "min_gain":   float(gain.min()),
            "E_P_gain":   float(P_A @ gain),
        })
    return records


# ---------------------------------------------------------------------------
# Figure: payoff distribution under arbitrage strategies
# ---------------------------------------------------------------------------

def make_figure(save_path: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Experiment 2 — Arbitrage Counterexamples", fontsize=13)

    # Panel A: Model A payoffs for h=1
    ax = axes[0]
    scenarios = ["Scenario 1\n(S: 1.0→1.2, p=0.6)", "Scenario 2\n(S: 1.0→1.1, p=0.4)"]
    gains = arbitrage_gain_A(1.0)
    bars = ax.bar(scenarios, gains, color=["steelblue", "steelblue"], edgecolor="black")
    ax.axhline(0, color="red", lw=1.5, ls="--", label="Zero gain")
    for bar, g in zip(bars, gains):
        ax.text(bar.get_x() + bar.get_width() / 2, g + 0.005,
                f"{g:.2f}", ha="center", va="bottom", fontweight="bold")
    ax.set_title("Model A: Gain with h=1\n(both gains > 0 → arbitrage)")
    ax.set_ylabel("Terminal Gain X_T")
    ax.set_ylim(-0.05, 0.35)
    ax.legend()

    # Panel B: robustness — E_P[gain] vs h
    ax = axes[1]
    h_vals = np.linspace(0, 10, 200)
    ep_gains = [float(P_A @ arbitrage_gain_A(h)) for h in h_vals]
    ax.plot(h_vals, ep_gains, "b-", lw=2, label=r"$E_P[h \cdot \Delta S]$")
    ax.axhline(0, color="red", lw=1.5, ls="--")
    ax.fill_between(h_vals, ep_gains, 0, where=[g > 0 for g in ep_gains],
                    alpha=0.2, color="green", label="Positive region")
    ax.set_title("Model A: Expected Gain vs h\n(positive for all h > 0)")
    ax.set_xlabel("h (holding)")
    ax.set_ylabel(r"$E_P[X_T]$")
    ax.legend()

    # Panel C: Model B — scatter of per-scenario gains for portfolio h=(1,0)
    ax = axes[2]
    h_B = np.array([1.0, 0.0])
    gains_B = dS_B @ h_B
    ax.bar(["Scenario 1\n(p=0.5)", "Scenario 2\n(p=0.5)"],
           gains_B, color="darkorange", edgecolor="black")
    ax.axhline(0, color="red", lw=1.5, ls="--")
    for i, g in enumerate(gains_B):
        ax.text(i, g + 0.005, f"{g:.3f}", ha="center", va="bottom", fontweight="bold")
    ax.set_title("Model B (d=2): Gain with h=(1,0)\n(asset 1, both scenarios > 0)")
    ax.set_ylabel("Terminal Gain")
    ax.set_ylim(-0.05, 0.45)

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

    print("=" * 60)
    print("EXP 2: Counterexample — Bounded Models with Arbitrage")
    print("=" * 60)

    # ---- Model A ----
    print("\n--- Model A: d=1, one-jump, both increments positive ---")
    print(f"  S0={S0_A}, S_T={S_T_A.tolist()}, P={P_A.tolist()}, ΔS={dS_A.tolist()}")

    print("\n  Step 2: Verify NA fails")
    na_A = check_na_model_A()
    print(f"    NA holds: {na_A['na_holds']}  (expect False)")
    print(f"    LP status: {na_A['lp_status']}")
    print(f"    Arbitrage h: {na_A['arb_h']}")
    print(f"    Gain on each scenario: {na_A['arb_gain']}")
    results["model_A_na"] = na_A

    print("\n  Step 3: Separating density via cvxpy (expect infeasible)")
    sep_A = try_separating_density_A()
    print(f"    cvxpy status: {sep_A['status']}  (expect infeasible)")
    results["model_A_sep_density"] = {"status": sep_A["status"]}

    print("\n  Step 4: Equivalent martingale measure via cvxpy (expect infeasible)")
    emm_A = try_emm_A()
    print(f"    cvxpy status: {emm_A['status']}  (expect infeasible)")
    results["model_A_emm"] = {"status": emm_A["status"]}

    # Analytical confirmation
    q1_analytic = (S0_A - S_T_A[1]) / (S_T_A[0] - S_T_A[1])
    print(f"\n  Analytical EMM check: q_1 = (S0 - S_T^2)/(S_T^1 - S_T^2) "
          f"= ({S0_A} - {S_T_A[1]})/({S_T_A[0]} - {S_T_A[1]}) = {q1_analytic:.4f}")
    print(f"  q_1 = {q1_analytic:.4f} < 0 → no valid probability measure")
    results["model_A_analytic_q1"] = float(q1_analytic)

    # ---- Robustness ----
    print("\n  Step 6: Robustness — multiple h values")
    robust = robustness_test_A()
    for r in robust:
        print(f"    h={r['h']:5.1f}: gain={r['gain']}, "
              f"min={r['min_gain']:.3f}, E_P={r['E_P_gain']:.4f}, is_arb={r['is_arb']}")
    results["model_A_robustness"] = robust

    # ---- Model B ----
    print("\n--- Model B: d=2, one-jump, both assets always positive ---")
    print(f"  S0={S0_B.tolist()}, ΔS=\n{dS_B}")

    na_B = check_na_model_B()
    print(f"\n  NA holds: {na_B['na_holds']}  (expect False)")
    print(f"  Arbitrage portfolio h: {na_B['arb_h']}")
    print(f"  Gain per scenario: {na_B['arb_gain']}")
    results["model_B_na"] = na_B

    emm_B = try_emm_B()
    print(f"\n  EMM cvxpy status: {emm_B['status']}  (expect infeasible)")
    results["model_B_emm"] = {"status": emm_B["status"]}

    # ---- Summary ----
    print("\n--- Summary ---")
    sep_infeasible = "infeasible" in sep_A["status"].lower()
    emm_A_infeasible = "infeasible" in emm_A["status"].lower()
    emm_B_infeasible = "infeasible" in emm_B["status"].lower()
    print(f"  Model A: NA=False, sep.density infeasible={sep_infeasible}, "
          f"EMM infeasible={emm_A_infeasible}")
    print(f"  Model B: NA=False, EMM infeasible={emm_B_infeasible}")
    results["summary"] = {
        "model_A_na":               False,
        "model_A_sep_infeasible":   sep_infeasible,
        "model_A_emm_infeasible":   emm_A_infeasible,
        "model_B_na":               False,
        "model_B_emm_infeasible":   emm_B_infeasible,
        "FTAP_duality_confirmed":   sep_infeasible and emm_A_infeasible,
    }

    # ---- Figure ----
    print("\n--- Saving figure ---")
    make_figure("/workspace/project/results/exp_2_payoffs.png")

    # ---- Save JSON ----
    out_path = "/workspace/project/results/exp_2_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved → {out_path}")
    print("\nEXP 2 COMPLETE.")


if __name__ == "__main__":
    main()
