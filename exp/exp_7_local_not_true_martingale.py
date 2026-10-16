"""
Experiment 7: Local Martingale but Not True Martingale.

Studies the "doubling process" on the state space E = {0} ∪ {2^k : k ≥ 0}:
  - S_0 = 1.
  - At each discrete time t_n, independently flip a fair coin:
      Heads (prob 0.5): S_{t_n} = 2 · S_{t_{n-1}}
      Tails (prob 0.5): S_{t_n} = 0,  and stays 0 forever.
  - t_n = T·(1 − 2^{−n}),  T = 1.0,  n = 1, …, N  (N=15).

Key results:
  1. E_P[S_{t_n}] = 1 for every n  (process is a P-martingale on each step).
  2. S_T → 0 P-a.s. as N → ∞  (P(S_T > 0) = (0.5)^N → 0).
  3. Stopped processes S^{σ_m} (σ_m = inf{t : S_t > 2^m}) are P-martingales
     on each localization.
  4. No equivalent TRUE martingale measure exists: if Q ~ P and S were a
     Q-true-martingale, then E_Q[S_T] = S_0 = 1, but S_T = 0 P-a.s.
     ⟹ S_T = 0 Q-a.s.  ⟹ E_Q[S_T] = 0 ≠ 1 — contradiction.
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import json
import os

os.makedirs('/workspace/project/results', exist_ok=True)

# ==============================================================================
# Parameters
# ==============================================================================
S0       = 1.0
T        = 1.0
N        = 15          # number of discrete steps
N_PATHS  = 10_000      # Monte Carlo paths
RNG_SEED = 42

t_n = np.array([T * (1.0 - 2.0**(-n)) for n in range(1, N + 1)])   # t_1, …, t_N
times = np.concatenate(([0.0], t_n))                                  # include t=0


# ==============================================================================
# Monte Carlo simulation
# ==============================================================================
rng   = np.random.default_rng(RNG_SEED)
# paths[i, n] = S_{t_n} for path i at step n  (column 0 = t=0)
paths = np.ones((N_PATHS, N + 1))                   # all start at 1.0

coins = rng.integers(0, 2, size=(N_PATHS, N))       # 1=heads, 0=tails

for n in range(N):
    alive              = paths[:, n] > 0.0
    heads              = coins[:, n] == 1
    # Double if alive and heads; zero if alive and tails; zero stays zero
    paths[alive & heads,  n + 1] = 2.0 * paths[alive & heads,  n]
    paths[alive & ~heads, n + 1] = 0.0
    paths[~alive,         n + 1] = 0.0


# ==============================================================================
# 1. Verify E_P[S_{t_n}] ≈ 1 for each n
# ==============================================================================
E_S_empirical  = paths.mean(axis=0)                 # shape (N+1,)
E_S_theoretical = np.ones(N + 1)                    # exactly 1 at every step


# ==============================================================================
# 2. Fraction of paths alive (S_{t_n} > 0) at each step
# ==============================================================================
frac_alive_empirical  = (paths > 0).mean(axis=0)
frac_alive_theory     = np.array([1.0] + [0.5**n for n in range(1, N + 1)])
# = P(all coins heads up to step n) = (0.5)^n


# ==============================================================================
# 3. Stopped processes S^{σ_m} for m = 1, 2, 3, 4
# ==============================================================================
# σ_m = inf{n : S_{t_n} > 2^m}.
# Once the running maximum exceeds 2^m, freeze the process at 2^m.
# (In practice: whenever paths[:, n] > 2^m, set paths[:, n] = 2^(m) and keep.)

def compute_stopped(paths: np.ndarray, m: int) -> np.ndarray:
    """
    Stopped process S^{σ_m} where σ_m = inf{n : S_n >= 2^m}.
    Once a path reaches 2^m (or 0, permanently), freeze it there.
    This gives a process bounded above by 2^m that is a true martingale.
    """
    bound   = float(2**m)
    stopped = paths.copy()
    for n in range(1, N + 1):
        # Freeze any path whose PREVIOUS stopped value already reached the bound.
        already_stopped = stopped[:, n - 1] >= bound
        stopped[already_stopped, n] = stopped[already_stopped, n - 1]
    return stopped


stopped_processes = {m: compute_stopped(paths, m) for m in [1, 2, 3, 4]}


# ==============================================================================
# 4. Verify martingale property for stopped processes
# ==============================================================================
def verify_stopped_martingale(stopped: np.ndarray, m: int) -> dict:
    """
    At each step n, for each group of paths with the same S^{σ_m}_{t_n} value v,
    check that the empirical E[S^{σ_m}_{t_{n+1}} | S^{σ_m}_{t_n} = v] ≈ v.
    Theoretical: if v < 2^m → 0.5·(2v) + 0.5·0 = v; if v = 2^m → stays = 2^m.
    """
    bound    = float(2**m)
    errors   = []
    for n in range(N):
        S_n   = stopped[:, n]
        S_n1  = stopped[:, n + 1]
        for v in np.unique(S_n):
            if v <= 0.0:
                continue
            mask       = S_n == v
            E_next_emp = float(S_n1[mask].mean())
            E_next_thr = v if v >= bound else 0.5 * 2 * v  # = v in both cases
            errors.append(abs(E_next_emp - E_next_thr))
    return {
        'max_conditional_error':  float(max(errors)) if errors else 0.0,
        'mean_conditional_error': float(np.mean(errors)) if errors else 0.0,
        'is_martingale':          bool(max(errors) < 0.1) if errors else True,
    }


martingale_checks = {
    f'm={m}': verify_stopped_martingale(stopped_processes[m], m)
    for m in [1, 2, 3, 4]
}


# ==============================================================================
# 5. Terminal values and "true martingale impossibility" proof
# ==============================================================================
terminal_values = paths[:, -1]
frac_terminal_alive  = float((terminal_values > 0).mean())
prob_alive_theory_N  = float(0.5**N)

# Formal proof (logical, not Monte Carlo)
proof = {
    'step_1': 'Suppose Q ~ P and S is a Q-true-martingale.',
    'step_2': f'Then E_Q[S_T] = S_0 = {S0} (martingale property).',
    'step_3': f'P(S_T > 0) = (0.5)^N → 0 as N → ∞.  For N={N}: (0.5)^{N} = {prob_alive_theory_N:.2e}.',
    'step_4': 'In the limit (N → ∞): S_T = 0  P-almost surely.',
    'step_5': 'Q ~ P implies Q(A) = 0 ⟺ P(A) = 0 for every event A.',
    'step_6': 'Therefore S_T = 0  Q-almost surely.',
    'step_7': 'But then E_Q[S_T] = 0 ≠ 1 = S_0 — contradiction!',
    'conclusion': 'No equivalent TRUE martingale measure exists.',
}


# ==============================================================================
# 6. Exact E_P[S_{t_n}] computation
# ==============================================================================
# At step n: P(S_{t_n} = 2^n) = (0.5)^n, P(S_{t_n} = 0) = 1 − (0.5)^n.
# E_P[S_{t_n}] = 2^n · (0.5)^n + 0 · (1 − (0.5)^n) = 1.0 for all n.
exact_expectations = {
    f'n={n}': {
        'P_alive':   float(0.5**n),
        'S_alive':   float(2.0**n),
        'E_P_exact': float(2.0**n * 0.5**n),   # = 1.0
    }
    for n in range(1, N + 1)
}


# ==============================================================================
# Assemble results
# ==============================================================================
results = {
    'parameters': {
        'S0': S0, 'T': T, 'N': N, 'n_paths': N_PATHS, 'seed': RNG_SEED,
    },
    'E_S_empirical': E_S_empirical.tolist(),
    'E_S_theoretical': E_S_theoretical.tolist(),
    'E_S_matches': bool(np.allclose(E_S_empirical, 1.0, atol=0.05)),
    'frac_alive_empirical': frac_alive_empirical.tolist(),
    'frac_alive_theoretical': frac_alive_theory.tolist(),
    'terminal_frac_alive': frac_terminal_alive,
    'prob_alive_theory_N': prob_alive_theory_N,
    'stopped_martingale_checks': martingale_checks,
    'all_stopped_are_martingales': bool(
        all(v['is_martingale'] for v in martingale_checks.values())
    ),
    'true_martingale_impossibility_proof': proof,
    'exact_expectations_per_step': exact_expectations,
}

with open('/workspace/project/results/exp_7_results.json', 'w') as f:
    json.dump(results, f, indent=2)


# ==============================================================================
# Figure: four-panel summary
# ==============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    'Experiment 7 — Doubling Process: Local Martingale ≠ True Martingale',
    fontsize=13, fontweight='bold',
)

# (a) Sample paths
ax = axes[0, 0]
n_show = min(60, N_PATHS)
t_plot = np.concatenate(([0.0], t_n))
for i in range(n_show):
    alive_idx = np.where(paths[i] > 0)[0]
    if len(alive_idx) > 0:
        last = alive_idx[-1]
        ax.step(t_plot[:last + 2],
                np.concatenate((paths[i, :last + 1], [0.0])),
                where='post', alpha=0.35, linewidth=0.9,
                color=plt.cm.tab20(i % 20))
    else:
        ax.step(t_plot, paths[i], where='post', alpha=0.2, linewidth=0.6, color='gray')
ax.set_xlabel('Time t', fontsize=10)
ax.set_ylabel('S(t)', fontsize=10)
ax.set_title(f'(a) Sample Paths  (showing {n_show}/{N_PATHS})', fontsize=10)
ax.set_yscale('log')
ax.set_ylim(bottom=0.5)
ax.axhline(S0, color='black', lw=1.5, linestyle='--', label='S₀=1')
ax.legend(fontsize=8)

# (b) E[S_{t_n}] vs n
ax = axes[0, 1]
steps = np.arange(N + 1)
ax.plot(steps, E_S_empirical,   'o-', color='steelblue', label='E_P[S_{t_n}] empirical', lw=2)
ax.plot(steps, E_S_theoretical, 'k--',                   label='Theoretical = 1.0', lw=1.5)
ax.fill_between(steps,
                E_S_empirical - 0.05, E_S_empirical + 0.05,
                alpha=0.15, color='steelblue', label='±0.05 band')
ax.set_xlabel('Step n', fontsize=10)
ax.set_ylabel('E[S_{t_n}]', fontsize=10)
ax.set_title('(b) Expected Value per Step\n(local martingale: E[S_{t_n}]=1 for all n)', fontsize=10)
ax.legend(fontsize=8)
ax.set_ylim(0.5, 1.5)

# (c) Fraction alive vs n
ax = axes[1, 0]
ax.semilogy(steps, frac_alive_empirical, 'o-', color='tomato',
            label='Empirical frac. alive', lw=2)
ax.semilogy(steps, frac_alive_theory, 'k--', label='(0.5)^n (theory)', lw=1.5)
ax.set_xlabel('Step n', fontsize=10)
ax.set_ylabel('P(S_{t_n} > 0)', fontsize=10)
ax.set_title('(c) Fraction of Paths Still Alive\n(→ 0 as n → ∞)', fontsize=10)
ax.legend(fontsize=8)

# (d) Histogram of terminal values
ax = axes[1, 1]
terminal_alive = terminal_values[terminal_values > 0]
if len(terminal_alive) > 0:
    ax.hist(np.log2(terminal_alive), bins=range(0, N + 2), color='mediumseagreen',
            edgecolor='black', alpha=0.8, label=f'log₂(S_T) for alive paths\n(n={len(terminal_alive)})')
    ax.set_xlabel('log₂(S_T)  for S_T > 0', fontsize=10)
else:
    ax.text(0.5, 0.5, 'No alive paths\n(S_T=0 everywhere)',
            ha='center', va='center', transform=ax.transAxes, fontsize=12)
ax.text(0.02, 0.95,
        f'P(S_T = 0) ≈ {1-frac_terminal_alive:.5f}\n'
        f'Theory:       {1 - prob_alive_theory_N:.5f}\n\n'
        f'True martingale\nimpossible:\n'
        f'E_Q[S_T]=0≠S_0=1\n(if Q~P)',
        transform=ax.transAxes, fontsize=8.5, va='top',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
ax.set_ylabel('Count', fontsize=10)
ax.set_title('(d) Terminal Value Distribution\n(concentrated at 0)', fontsize=10)
if len(terminal_alive) > 0:
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('/workspace/project/results/exp_7_doubling.png', dpi=120, bbox_inches='tight')
plt.close()


# ==============================================================================
# Console summary
# ==============================================================================
print("=" * 60)
print("Experiment 7: Local Martingale but Not True Martingale")
print("=" * 60)
print(f"\n  N = {N} steps,  n_paths = {N_PATHS},  seed = {RNG_SEED}")
print(f"  t_n = T·(1 - 2^(-n)) for n=1,...,{N}")

print(f"\n  E_P[S_{{t_n}}] (should ≈ 1.0):")
for n in [0, 1, 5, 10, 14, 15]:
    print(f"    n={n:2d}: empirical={E_S_empirical[n]:.4f}, theory=1.0")

print(f"\n  Fraction alive at step N={N}: {frac_alive_empirical[-1]:.6f}")
print(f"  Theory (0.5)^{N} = {prob_alive_theory_N:.2e}")
print(f"  Terminal S_T > 0: {frac_terminal_alive:.6f}")

print(f"\n  Stopped process martingale checks:")
for key, val in martingale_checks.items():
    print(f"    {key}: max_error={val['max_conditional_error']:.4f},"
          f" is_martingale={val['is_martingale']}")

print(f"\n  Impossibility proof:")
for k, v in proof.items():
    print(f"    {k}: {v}")

print("\nResults saved to /workspace/project/results/exp_7_results.json")
print("Figure  saved to /workspace/project/results/exp_7_doubling.png")
