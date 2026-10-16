"""
Experiment 5: Constructive Separation Density and Equivalent Local Martingale Measure.

Builds two locally-bounded stopped models for a 2-step binomial price process,
finds a separating density Z for each via cvxpy, defines Q = Z*P, verifies Q~P
and the martingale property, and concludes that the original process is a
Q-local martingale.  Also verifies the reverse direction: given an EMM, all
admissible gains have zero Q-expectation (NA holds).
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import json
import os

os.makedirs('/workspace/project/results', exist_ok=True)

# ==============================================================================
# Scenario tree (d=1, T=2, S0=1.0, fair coin p_up=0.5)
# ==============================================================================
# Four terminal scenarios: uu, ud, du, dd  (each with probability 0.25)
# t=0 : S = 1.0
# t=1 : 1.5 (scenarios 0,1)  |  0.7 (scenarios 2,3)
# t=2 : see models below

SCENARIOS = ['uu', 'ud', 'du', 'dd']
S0  = 1.0
S1  = np.array([1.5, 1.5, 0.7, 0.7])          # values at t=1
P   = np.full(4, 0.25)                          # physical measure (fair coin)

# ---- Localization 1 (sigma_1, bound B1=2) ------------------------------------
# In the uu path the process would reach 2.25 > 2; the stopped process freezes
# at the last value before crossing, giving S2_stopped[uu] = 2.0 (at the bound).
S2_stopped = np.array([2.0,  1.05, 1.05, 0.49])
B1 = 2.0

# ---- Localization 2 (sigma_2, bound B2=4, no stopping in this tree) ----------
# All t=2 values lie below 4, so the stopped process equals the original.
S2_full    = np.array([2.25, 1.05, 1.05, 0.49])
B2 = 4.0


# ==============================================================================
# Build admissible gain matrix  G  (n_scenarios × n_strategies)
# ==============================================================================
# Three orthogonal strategy directions:
#   H_0    : position held t=0 → t=1  (affects all 4 scenarios)
#   H_up   : position held t=1 → t=2 in the up subtree  (scenarios 0,1)
#   H_down : position held t=1 → t=2 in the down subtree (scenarios 2,3)
# Terminal gain for scenario k = H_0*(S1_k - S0) + H_up/down*(S2_k - S1_k)

def build_gain_matrix(S2: np.ndarray) -> np.ndarray:
    G = np.zeros((4, 3))
    G[:, 0] = S1 - S0                           # H_0 contribution
    G[0, 1] = S2[0] - S1[0]                     # H_up: uu
    G[1, 1] = S2[1] - S1[1]                     # H_up: ud
    G[2, 2] = S2[2] - S1[2]                     # H_down: du
    G[3, 2] = S2[3] - S1[3]                     # H_down: dd
    return G


# ==============================================================================
# Find separating density via cvxpy
# ==============================================================================
# Conditions:
#   z_k >= eps                        (strict positivity → Q ~ P)
#   sum_k P_k z_k = 1                (normalization: E_P[Z] = 1)
#   sum_k P_k z_k G_{k,j} = 0  ∀j   (martingale condition = separation)

def solve_separating_density(P: np.ndarray, G: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    n = len(P)
    z = cp.Variable(n)
    constraints = [z >= eps, P @ z == 1.0]
    for j in range(G.shape[1]):
        constraints.append(P @ cp.multiply(z, G[:, j]) == 0)
    prob = cp.Problem(cp.Minimize(0), constraints)
    prob.solve(solver=cp.SCS, verbose=False)
    if z.value is None:
        raise RuntimeError("cvxpy did not find a feasible solution.")
    return z.value


G_stopped = build_gain_matrix(S2_stopped)
G_full    = build_gain_matrix(S2_full)

Z1 = solve_separating_density(P, G_stopped)
Z2 = solve_separating_density(P, G_full)

Q1 = P * Z1          # ELMM for stopped model
Q2 = P * Z2          # ELMM for full model (also EMM for original S)


# ==============================================================================
# Verify martingale property under each Q
# ==============================================================================

def verify_martingale(Q: np.ndarray, S2: np.ndarray) -> dict:
    # E_Q[S1] = S0
    E_S1 = float(np.sum(Q * S1))
    # E_Q[S2 | S1=1.5]
    up_mask   = (S1 == 1.5)
    Q_up_tot  = Q[up_mask].sum()
    E_S2_up   = float(np.sum(Q[up_mask] * S2[up_mask]) / Q_up_tot)
    # E_Q[S2 | S1=0.7]
    dn_mask   = (S1 == 0.7)
    Q_dn_tot  = Q[dn_mask].sum()
    E_S2_dn   = float(np.sum(Q[dn_mask] * S2[dn_mask]) / Q_dn_tot)
    return {
        'E_Q_S1':            E_S1,
        'S0':                float(S0),
        'martingale_t1':     bool(abs(E_S1 - S0) < 1e-4),
        'E_Q_S2_given_up':   E_S2_up,
        'node_up':           1.5,
        'martingale_t2_up':  bool(abs(E_S2_up - 1.5) < 1e-4),
        'E_Q_S2_given_down': E_S2_dn,
        'node_down':         0.7,
        'martingale_t2_down': bool(abs(E_S2_dn - 0.7) < 1e-4),
    }


mart1 = verify_martingale(Q1, S2_stopped)
mart2 = verify_martingale(Q2, S2_full)


# ==============================================================================
# Q ~ P verification
# ==============================================================================
Q1_equiv_P = bool(np.all(Q1 > 1e-7))
Q2_equiv_P = bool(np.all(Q2 > 1e-7))


# ==============================================================================
# Reverse direction: given EMM, all admissible gains have zero Q-expectation
# ==============================================================================
# Any strategy H = (h0, h_up, h_dn) gives gain G @ H.
# E_Q[G @ H] = (Q.T @ G) @ H = 0 @ H = 0 (since Q satisfies martingale eqs).

def test_na_direction(Q: np.ndarray, G: np.ndarray) -> dict:
    test_strategies = {
        'buy_and_hold':  np.array([1.0, 0.0, 0.0]),
        'up_subtree':    np.array([0.0, 1.0, 0.0]),
        'down_subtree':  np.array([0.0, 0.0, 1.0]),
        'mixed':         np.array([1.0, 0.5, -0.5]),
    }
    results = {}
    for name, h in test_strategies.items():
        gain = G @ h
        E_Q_gain = float(np.sum(Q * gain))
        results[name] = {'E_Q_gain': E_Q_gain, 'zero': bool(abs(E_Q_gain) < 1e-4)}
    return results


na1 = test_na_direction(Q1, G_stopped)
na2 = test_na_direction(Q2, G_full)


# ==============================================================================
# Assemble results
# ==============================================================================
results = {
    'model_stopped_B2': {
        'bound':           B1,
        'S2':              S2_stopped.tolist(),
        'P':               P.tolist(),
        'Z':               Z1.tolist(),
        'Q':               Q1.tolist(),
        'Q_equiv_P':       Q1_equiv_P,
        'Z_sum_check':     float(np.sum(P * Z1)),
        'martingale':      mart1,
        'na_direction':    na1,
    },
    'model_full_B4': {
        'bound':           B2,
        'S2':              S2_full.tolist(),
        'P':               P.tolist(),
        'Z':               Z2.tolist(),
        'Q':               Q2.tolist(),
        'Q_equiv_P':       Q2_equiv_P,
        'Z_sum_check':     float(np.sum(P * Z2)),
        'martingale':      mart2,
        'na_direction':    na2,
    },
    'local_martingale_conclusion': (
        'S is a Q2-local martingale (in fact a Q2-true martingale) via the '
        'sequence sigma_1 (bound=2) and sigma_2=T (bound=4). Each stopped '
        'process S^{sigma_m} is a Q_m-martingale, confirming the FTAP for '
        'locally bounded processes.'
    ),
}

with open('/workspace/project/results/exp_5_results.json', 'w') as f:
    json.dump(results, f, indent=2)


# ==============================================================================
# Figure: localization structure
# ==============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 7))

specs = [
    ('Localization 1  (bound B=2, stopped model)', S2_stopped, Q1, Z1, B1),
    ('Localization 2  (bound B=4, full model)',    S2_full,    Q2, Z2, B2),
]

# Layout positions (x=time, y=vertical spread for readability)
node_pos = {
    't0':   (0.0, 0.50),
    't1_u': (1.0, 0.72),
    't1_d': (1.0, 0.28),
    't2_uu': (2.0, 0.88),
    't2_ud': (2.0, 0.58),
    't2_du': (2.0, 0.42),
    't2_dd': (2.0, 0.12),
}

for ax, (title, S2, Q, Z, B) in zip(axes, specs):
    # Branches
    edges = [
        ('t0', 't1_u'), ('t0', 't1_d'),
        ('t1_u', 't2_uu'), ('t1_u', 't2_ud'),
        ('t1_d', 't2_du'), ('t1_d', 't2_dd'),
    ]
    colors = ['steelblue', 'tomato', 'steelblue', 'tomato', 'steelblue', 'tomato']
    for (src, dst), col in zip(edges, colors):
        x0, y0 = node_pos[src]; x1, y1 = node_pos[dst]
        ax.plot([x0, x1], [y0, y1], color=col, linewidth=2, zorder=1)

    # Nodes and labels
    node_data = [
        ('t0',    S0,      None,         'gold'),
        ('t1_u',  1.5,     None,         'skyblue'),
        ('t1_d',  0.7,     None,         'skyblue'),
        ('t2_uu', S2[0],   (Q[0], Z[0]), 'lightgreen'),
        ('t2_ud', S2[1],   (Q[1], Z[1]), 'lightgreen'),
        ('t2_du', S2[2],   (Q[2], Z[2]), 'lightgreen'),
        ('t2_dd', S2[3],   (Q[3], Z[3]), 'lightgreen'),
    ]
    for key, val, qz, col in node_data:
        x, y = node_pos[key]
        ax.scatter([x], [y], s=320, c=col, zorder=3, edgecolors='black', linewidths=0.8)
        label = f'S={val:.2f}'
        if qz is not None:
            label += f'\nQ={qz[0]:.3f}\nZ={qz[1]:.3f}'
        ax.annotate(label, (x, y), fontsize=7.5,
                    xytext=(8 if x < 1.5 else 6, 4),
                    textcoords='offset points', zorder=4)

    # Axis formatting
    ax.set_xlim(-0.4, 2.9)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['t = 0', 't = 1', 't = 2'], fontsize=11)
    ax.set_yticks([])
    ax.set_title(title, fontsize=11, fontweight='bold', pad=8)
    ax.axvline(x=0, color='gray', linestyle=':', alpha=0.4)
    ax.axvline(x=1, color='gray', linestyle=':', alpha=0.4)
    ax.axvline(x=2, color='gray', linestyle=':', alpha=0.4)

    # Bound annotation
    ax.text(2.82, 0.95, f'Bound\nB = {B}', fontsize=9, color='darkred',
            ha='center', va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='mistyrose', alpha=0.8))

    # Legend snippet
    mart = mart1 if B == B1 else mart2
    summary = (f"Q~P: {Q.min():.4f} > 0\n"
               f"E_Q[S1]={mart['E_Q_S1']:.4f}≈S0={S0}\n"
               f"E_Q[S2|up]={mart['E_Q_S2_given_up']:.4f}≈1.5\n"
               f"E_Q[S2|dn]={mart['E_Q_S2_given_down']:.4f}≈0.7")
    ax.text(0.02, 0.02, summary, transform=ax.transAxes, fontsize=8,
            verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

fig.suptitle(
    'Experiment 5 — Constructive ELMM via Localization\n'
    'Each stopped process is a Q-martingale  ⟹  S is a Q-local martingale',
    fontsize=12, fontweight='bold', y=1.01,
)
plt.tight_layout()
plt.savefig('/workspace/project/results/exp_5_localization.png', dpi=120, bbox_inches='tight')
plt.close()

# ==============================================================================
# Console summary
# ==============================================================================
print("=" * 60)
print("Experiment 5: Local Martingale Measure via Localization")
print("=" * 60)
for label, Q, Z, mart, B in [
    ('Stopped (B=2)', Q1, Z1, mart1, B1),
    ('Full    (B=4)', Q2, Z2, mart2, B2),
]:
    print(f"\n  Model {label}:")
    print(f"    Z   = {np.round(Z, 5)}")
    print(f"    Q   = {np.round(Q, 5)}")
    print(f"    Q~P = {bool(np.all(Q > 0))}")
    print(f"    E_Q[S1] = {mart['E_Q_S1']:.6f}  (S0={S0})")
    print(f"    E_Q[S2|up]  = {mart['E_Q_S2_given_up']:.6f}  (node=1.5)")
    print(f"    E_Q[S2|dn]  = {mart['E_Q_S2_given_down']:.6f}  (node=0.7)")
    all_ok = mart['martingale_t1'] and mart['martingale_t2_up'] and mart['martingale_t2_down']
    print(f"    Martingale property holds: {all_ok}")

print("\nResults saved to /workspace/project/results/exp_5_results.json")
print("Figure  saved to /workspace/project/results/exp_5_localization.png")
