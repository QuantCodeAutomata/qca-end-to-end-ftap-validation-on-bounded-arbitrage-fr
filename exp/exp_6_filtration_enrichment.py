"""
Experiment 6: Continuous-Time Information Between Price Jumps.

Demonstrates the role of enriched filtrations in arbitrage theory.
A signal I revealed at t=0.5 (before the price jump at t=1.0) allows a
full-filtration strategy to achieve strictly positive gains in every scenario —
an arbitrage that is invisible to the price-history-only filtration.
The price-only model is arbitrage-free and possesses a unique EMM.
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

os.makedirs('/workspace/project/results', exist_ok=True)

# ==============================================================================
# Model definition
# ==============================================================================
# Time points: 0.0, 0.5, 1.0 (T=1.0)
# At t=0.5: signal I is revealed.  Price stays at S=1.0.
# At t=1.0: price jumps depending on I.
#
# Scenario omega_1: I=1  → S_1 = 1.3  (delta_S = +0.3)
# Scenario omega_2: I=0  → S_1 = 0.8  (delta_S = −0.2)

T       = 1.0
S0      = 1.0
times   = np.array([0.0, 0.5, 1.0])

scenarios  = ['omega_1 (I=1)', 'omega_2 (I=0)']
P          = np.array([0.5, 0.5])          # physical measure

I_values   = np.array([1, 0])              # signal at t=0.5
S_at_05    = np.array([1.0, 1.0])          # price at t=0.5 (same in both)
S_at_10    = np.array([1.3, 0.8])          # price at t=1.0
delta_S    = S_at_10 - S_at_05            # [+0.3, -0.2]


# ==============================================================================
# Strategy classes
# ==============================================================================

# ---- Full-filtration strategy ------------------------------------------------
# H_full(t=1-, omega) is set AFTER seeing I at t=0.5 but BEFORE the jump.
# Strategy: buy if I=1, sell if I=0.
H_full = np.array([+1.0, -1.0])
gains_full = H_full * delta_S             # [+0.3, +0.2]  — both positive!

# ---- Price-history-only strategy --------------------------------------------
# At t=0.5+, both scenarios share the same price history {S_0=1.0, S_{0.5}=1.0}.
# A price-history-measurable strategy must assign the SAME holding to both.
# Best constant strategy: h such that max(h*delta_S) > 0.
# Check for arbitrage: need h*0.3 ≥ 0 AND h*(−0.2) ≥ 0 → h=0 only.
H_price_candidates = [1.0, -1.0, 0.0, 0.4]
price_strategy_results = {}
for h in H_price_candidates:
    gains = h * delta_S
    is_arb = bool(np.all(gains >= 0) and np.any(gains > 0))
    price_strategy_results[f'h={h}'] = {
        'gains':       gains.tolist(),
        'is_arbitrage': is_arb,
    }

# ---- Neutral strategy (same H regardless of signal) -------------------------
H_neutral  = np.array([0.4, 0.4])
gains_neutral = H_neutral * delta_S       # [+0.12, -0.08]  — not all positive


# ==============================================================================
# Equivalent Martingale Measure for the price-only model
# ==============================================================================
# Need Q = (q, 1-q), q > 0, such that E_Q[S_1] = S_0:
#   q * 1.3 + (1-q) * 0.8 = 1.0
#   0.5*q = 0.2  →  q = 0.4
q_emm = (S0 - S_at_10[1]) / (S_at_10[0] - S_at_10[1])   # = 0.4
Q_emm = np.array([q_emm, 1.0 - q_emm])

# Verify
E_Q_S1 = float(np.sum(Q_emm * S_at_10))
emm_valid = bool(abs(E_Q_S1 - S0) < 1e-10 and np.all(Q_emm > 0))

# Separating density
Z_emm = Q_emm / P                           # [0.8, 1.2]
Z_sum  = float(np.sum(P * Z_emm))           # = 1.0

# Verify separation: E_Q[H_price * delta_S] = 0 for any constant H
# E_Q[delta_S] = q*0.3 + (1-q)*(-0.2) = 0.4*0.3 + 0.6*(-0.2) = 0.12 - 0.12 = 0.0
E_Q_delta_S = float(np.sum(Q_emm * delta_S))


# ==============================================================================
# Full-filtration arbitrage check
# ==============================================================================
# Under ANY Q~P (q in (0,1)), E_Q[gains_full]:
#   = q*0.3 + (1-q)*0.2  > 0  for ALL q in (0,1)
# So the full-filtration strategy cannot be separated → NA fails in F^full.

def check_full_filtration_separation(q_values: np.ndarray) -> dict:
    results = {}
    for q in q_values:
        Q_test = np.array([q, 1 - q])
        E_val  = float(np.sum(Q_test * gains_full))
        results[f'q={q:.2f}'] = {'E_Q_gains_full': E_val, 'positive': bool(E_val > 0)}
    return results

q_test_vals = np.linspace(0.05, 0.95, 10)
sep_check   = check_full_filtration_separation(q_test_vals)

# ==============================================================================
# Predictability demonstration
# ==============================================================================
# H_full is measurable w.r.t. F_{0.5} (which includes I).
# H_price is measurable w.r.t. the natural price filtration F^S_{0.5-},
# which cannot distinguish omega_1 from omega_2 at t=0.5.

price_history_distinguishes = {
    't=0':  {'S_omega1': float(S0), 'S_omega2': float(S0),
             'distinguishes': False},
    't=0.5':{'S_omega1': float(S_at_05[0]), 'S_omega2': float(S_at_05[1]),
             'distinguishes': False},      # same price in both scenarios
    't=1.0':{'S_omega1': float(S_at_10[0]), 'S_omega2': float(S_at_10[1]),
             'distinguishes': True},       # prices differ AFTER the jump
}

enriched_filtration_distinguishes = {
    't=0':  {'I_omega1': int(I_values[0]), 'I_omega2': int(I_values[1]),
             'distinguishes': False},      # I not yet revealed
    't=0.5':{'I_omega1': int(I_values[0]), 'I_omega2': int(I_values[1]),
             'distinguishes': True},       # I revealed → strategies can differ
}


# ==============================================================================
# Assemble results
# ==============================================================================
results = {
    'model': {
        'T': T,
        'S0': S0,
        'times': times.tolist(),
        'scenarios': scenarios,
        'P': P.tolist(),
        'I_values': I_values.tolist(),
        'S_at_0.5': S_at_05.tolist(),
        'S_at_1.0': S_at_10.tolist(),
        'delta_S':  delta_S.tolist(),
    },
    'full_filtration_strategy': {
        'H_full':     H_full.tolist(),
        'gains_full': gains_full.tolist(),
        'is_arbitrage': bool(np.all(gains_full > 0)),
        'separation_fails_for_all_q': sep_check,
    },
    'price_only_strategies': {
        'candidates': price_strategy_results,
        'no_price_only_arbitrage': True,
        'comment': 'h*delta_S >= 0 in both scenarios requires h=0 (no profit).',
    },
    'neutral_strategy': {
        'H_neutral':     H_neutral.tolist(),
        'gains_neutral': gains_neutral.tolist(),
        'is_arbitrage':  bool(np.all(gains_neutral >= 0) and np.any(gains_neutral > 0)),
    },
    'EMM_price_only': {
        'q_emm':        float(q_emm),
        'Q':            Q_emm.tolist(),
        'Z':            Z_emm.tolist(),
        'Z_sum':        Z_sum,
        'Q_equiv_P':    bool(np.all(Q_emm > 0)),
        'E_Q_S1':       E_Q_S1,
        'martingale':   emm_valid,
        'E_Q_delta_S':  E_Q_delta_S,
    },
    'filtration_comparison': {
        'price_history': price_history_distinguishes,
        'enriched':      enriched_filtration_distinguishes,
        'conclusion': (
            'The two scenarios share the same price history up to t=0.5 '
            '(price=1.0 in both), so a price-only strategy cannot distinguish '
            'them before the jump.  The enriched filtration reveals I at t=0.5, '
            'allowing H_full to exploit the signal — creating arbitrage not '
            'present in the price-only filtration.'
        ),
    },
}

with open('/workspace/project/results/exp_6_results.json', 'w') as f:
    json.dump(results, f, indent=2)


# ==============================================================================
# Figure: filtration tree with signal and price events
# ==============================================================================
fig, axes = plt.subplots(1, 2, figsize=(15, 7))

# ---- Left panel: filtration tree -------------------------------------------
ax = axes[0]
ax.set_xlim(-0.3, 1.5)
ax.set_ylim(-0.15, 1.15)
ax.set_title('Enriched Filtration Tree\n(signal I revealed at t=0.5)', fontweight='bold')

# t=0 node
ax.scatter([0], [0.5], s=400, c='gold', zorder=5, edgecolors='black')
ax.text(0, 0.5, ' t=0\n S=1.0', fontsize=9, va='center')

# t=0.5 nodes (price same, but signal splits)
ax.scatter([0.5, 0.5], [0.75, 0.25], s=400, c='skyblue', zorder=5, edgecolors='black')
ax.text(0.5, 0.75, ' I=1\n S=1.0', fontsize=9, va='center', color='steelblue')
ax.text(0.5, 0.25, ' I=0\n S=1.0', fontsize=9, va='center', color='tomato')
ax.plot([0, 0.5], [0.5, 0.75], 'steelblue', lw=2, label='I=1 (prob 0.5)')
ax.plot([0, 0.5], [0.5, 0.25], 'tomato',    lw=2, label='I=0 (prob 0.5)')
ax.text(0.15, 0.67, 'p=0.5', fontsize=8, color='steelblue')
ax.text(0.15, 0.32, 'p=0.5', fontsize=8, color='tomato')

# t=1.0 nodes (price jumps)
ax.scatter([1.0, 1.0], [0.75, 0.25], s=400, c='lightgreen', zorder=5, edgecolors='black')
ax.text(1.0, 0.75, '  S=1.3\n  (ω₁)', fontsize=9, va='center', color='steelblue')
ax.text(1.0, 0.25, '  S=0.8\n  (ω₂)', fontsize=9, va='center', color='tomato')
ax.plot([0.5, 1.0], [0.75, 0.75], 'steelblue', lw=2)
ax.plot([0.5, 1.0], [0.25, 0.25], 'tomato',    lw=2)
ax.annotate('ΔS=+0.3', (0.75, 0.75), xytext=(0, 8), textcoords='offset points',
            fontsize=8, color='steelblue', ha='center')
ax.annotate('ΔS=−0.2', (0.75, 0.25), xytext=(0, -14), textcoords='offset points',
            fontsize=8, color='tomato', ha='center')

# Time axis
for t, label in zip([0, 0.5, 1.0], ['t=0', 't=0.5\n(I revealed)', 't=1.0\n(jump)']):
    ax.axvline(x=t, color='gray', linestyle=':', alpha=0.5)
    ax.text(t, -0.08, label, ha='center', fontsize=8.5, color='gray')

ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel('')
leg = ax.legend(loc='upper right', fontsize=8)

# ---- Right panel: strategy gains comparison --------------------------------
ax2 = axes[1]
ax2.set_title('Strategy Gains Comparison\n(full-filtration vs price-only)', fontweight='bold')

x  = np.arange(2)
w  = 0.25
sc = ['ω₁ (I=1)', 'ω₂ (I=0)']

bars1 = ax2.bar(x - w,    gains_full,    width=w, label='H_full (+1,−1)',
                color=['steelblue', 'steelblue'], edgecolor='black', alpha=0.85)
bars2 = ax2.bar(x,        gains_neutral, width=w, label='H_neutral (0.4,0.4)',
                color=['orange', 'orange'],       edgecolor='black', alpha=0.85)
# Price-only with h=1
gains_price_h1 = np.array([1.0, 1.0]) * delta_S
bars3 = ax2.bar(x + w,    gains_price_h1, width=w, label='H_price h=1.0 (1,1)',
                color=['gray', 'gray'],           edgecolor='black', alpha=0.85)

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h + (0.005 if h >= 0 else -0.015),
                 f'{h:.2f}', ha='center', va='bottom' if h >= 0 else 'top', fontsize=8)

ax2.axhline(0, color='black', lw=1)
ax2.axhspan(0.001, 0.5, alpha=0.07, color='green', label='Arbitrage region (gain>0)')
ax2.set_xticks(x)
ax2.set_xticklabels(sc, fontsize=11)
ax2.set_ylabel('Terminal Gain', fontsize=11)
ax2.legend(fontsize=8, loc='lower right')
ax2.set_ylim(-0.35, 0.45)

# Annotation: H_full is arbitrage
ax2.annotate('✓ H_full: BOTH gains > 0\n   (arbitrage in F^full)',
             xy=(-0.125, 0.22), xytext=(0.3, 0.38),
             fontsize=8.5, color='darkgreen',
             arrowprops=dict(arrowstyle='->', color='darkgreen', lw=1.3))

# EMM info
emm_txt = (f'EMM (price-only): q={q_emm:.1f}\n'
           f'E_Q[ΔS] = {E_Q_delta_S:.4f} ≈ 0\n'
           f'Q = ({Q_emm[0]:.1f}, {Q_emm[1]:.1f}), Q~P ✓')
ax2.text(0.98, 0.98, emm_txt, transform=ax2.transAxes, fontsize=8,
         ha='right', va='top', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

plt.suptitle(
    'Experiment 6 — Filtration Enrichment and Arbitrage\n'
    'Price-only model is NA;  enriched filtration (with signal I) admits arbitrage',
    fontsize=11, fontweight='bold',
)
plt.tight_layout()
plt.savefig('/workspace/project/results/exp_6_filtration.png', dpi=120, bbox_inches='tight')
plt.close()

# ==============================================================================
# Console summary
# ==============================================================================
print("=" * 60)
print("Experiment 6: Filtration Enrichment")
print("=" * 60)
print(f"\n  Scenarios: {scenarios}")
print(f"  P = {P},  S_0 = {S0}")
print(f"  delta_S = {delta_S}  (at t=1.0)")

print(f"\n  Full-filtration strategy H_full = {H_full.tolist()}")
print(f"  Gains = {gains_full.tolist()} — both POSITIVE → arbitrage in F^full")

print(f"\n  Price-only: any constant h gives gains h*{delta_S.tolist()}")
print(f"  Arbitrage requires h*0.3≥0 and h*(-0.2)≥0 → h=0 only (no arbitrage)")

print(f"\n  EMM (price-only): q = {q_emm:.4f}")
print(f"  Q = {Q_emm.tolist()},  Z = {Z_emm.tolist()}")
print(f"  E_Q[S_1] = {E_Q_S1:.6f} = S_0 = {S0}  (martingale ✓)")
print(f"  E_Q[delta_S] = {E_Q_delta_S:.4f}  (separation ✓)")

print("\nResults saved to /workspace/project/results/exp_6_results.json")
print("Figure  saved to /workspace/project/results/exp_6_filtration.png")
