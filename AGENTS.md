# FTAP Validation Repository — Agent Notes

## Project overview
Numerical validation of the Fundamental Theorem of Asset Pricing (FTAP) using
discrete/finite probability spaces and self-contained Python experiment scripts.

## Directory layout
```
exp/          Standalone experiment scripts (exp_1 … exp_7)
tests/        pytest test suite (test_exp_1 … test_exp_7)
results/      JSON result files and PNG figures (auto-created by experiments)
src/          Core library modules (see below)
data/         (empty)
```

## src/ module inventory (implemented 2025)

| Module | Key classes / functions |
|--------|------------------------|
| `scenario_tree.py` | `ScenarioNode`, `ScenarioTree`, `JumpEvent`, `PredictableStrategy`, `GainProcess`, `AdmissibilityChecker`; factories `build_binary_tree_1d`, `build_simple_2d_tree`, `build_arbitrage_tree` |
| `geometry_lemma.py` | `compute_u_J` (SLSQP), `compute_J0`, `compute_projection` (SVD), `compute_beta_J` (random-restart SLSQP), `compute_kappa_J`, `compute_all_geometry`, `project_strategy`, `compute_R_process`, `compute_V_process`, `verify_lemma3_inequalities` |
| `separation.py` | `solve_separating_density` (cvxpy CLARABEL), `build_claim_matrix`, `solve_martingale_measure` (linear martingale constraints via cvxpy), `verify_martingale_property`, `verify_equivalence` |
| `compactness.py` | `compute_mu_norm_squared`, `compute_total_mu_norm_squared`, `forward_convex_average`, `compute_gain_sup_residual` |
| `local_martingale_example.py` | `generate_doubling_process_paths`, `compute_stopping_times_sigma_m`, `compute_stopped_process`, `verify_stopped_martingale`, `verify_terminal_zero`, `check_true_martingale_impossible` |
| `filtration_enrichment.py` | `EventNode`, `EnrichedFiltrationTree`, `FullFiltrationStrategy`, `PriceHistoryOnlyStrategy`, `build_enriched_filtration_example`, `compute_gains_enriched`, `compare_strategy_outcomes` |

### src/ design conventions
- `ScenarioTree` uses **non-recombining** trees: each node has a unique ID encoding its full path (e.g. `root_u_d_u`).
- **Predictable strategy**: `PredictableStrategy.holdings[node_id]` is the holding decided at `node_id` and applied to the jump from that node to its children (F_{t−}-measurable).
- **GainProcess**: `(H·S)_t = Σ_{j=0}^{i-1} ⟨H_{path[j]}, path[j+1].state − path[j].state⟩`, gain[0]=0.
- **Martingale constraint in `solve_martingale_measure`**: for each internal node n with children c_j, the constraint `Σ_j (Σ_{k∈subtree(c_j)} Q_k) · (S_{c_j}[d] − S_n[d]) = 0` is **linear** in terminal Q variables → solved with CLARABEL.
- **`compute_u_J`**: uses SLSQP with 11 random restarts; when J has both positive and negative elements in the same dimension, u_J = 0 (correct: no separating direction exists).
- **`compute_beta_J`**: uses 20+ random restarts on the unit sphere in span(J0) to find global minimum of `max_k(−⟨v, z_k⟩)`.
- **`FullFiltrationStrategy`**: composite key `"{node_id}_s{signal}"` stores signal-dependent holdings; `get_holding` traces ancestry to find the most recent signal_arrival node.
- **Doubling process**: `t_n = T·(1 − 2^{−n})`, grid accumulates toward T; S doubles with prob 0.5 or collapses to 0 (absorbing). With N=15, virtually all paths reach S_T=0.

## Dependencies
numpy, scipy, cvxpy, matplotlib, pytest — all pre-installed.

## Running experiments
```bash
python exp/exp_N_name.py          # standalone; writes to results/
pytest tests/ -v                  # 32 tests, all should pass
```

## Key design rules
- **Self-contained**: every exp_*.py uses only numpy/scipy/cvxpy/matplotlib/json/os.
- **results/ auto-created**: each experiment calls `os.makedirs('…/results', exist_ok=True)`.
- **Agg backend**: every exp file starts with `matplotlib.use('Agg')`.

## Experiment summaries

| File | Topic | Key output |
|------|-------|-----------|
| exp_1 | FTAP validation: 2-step binomial (d=1), NA↔EMM, Q-martingale property | exp_1_results.json, exp_1_tree.png |
| exp_2 | Counterexample: one-jump models where all ΔS>0 → NA fails, cvxpy infeasible | exp_2_results.json, exp_2_payoffs.png |
| exp_3 | Geometry lemma: d=2 redundant (S=(Y,Y)), projection, per-step κ, Lemma 3 bounds | exp_3_results.json, exp_3_pathwise.png |
| exp_4 | L²(μ) compactness: 3-step binomial, 5 strategies, forward tail averages, cvxpy QP | exp_4_results.json, exp_4_convergence.png |
| exp_5 | Local martingale measure via localization (2-step binomial, 2 stopped models) | exp_5_results.json, exp_5_localization.png |
| exp_6 | Filtration enrichment — signal I creates arbitrage in F^full but not price filtration | exp_6_results.json, exp_6_filtration.png |
| exp_7 | Doubling process: local martingale but no equivalent TRUE martingale measure | exp_7_results.json, exp_7_doubling.png |

## Critical implementation detail — exp_7 stopped process
`compute_stopped` must use `stopped[:, n-1] >= bound` (freeze PREVIOUS step
that reached the bound), NOT `stopped[:, n] > bound` (freeze current step if
it overshoots). The latter breaks the martingale property at the boundary.

Correct: `sigma_m = inf{n : S_n >= 2^m}`.  Then S^{σ_m}_n = S_{n ∧ σ_m}.
For the doubling process, paths that reach 2^m stay frozen there forever;
paths that die (go to 0) stay at 0.  Both give E[next | current] = current ✓.

## EMM formulas (binomial models)
- 1-step: `q = (S0 - down)/(up - down)`
- 2-step binomial (up=1.2, down=0.8, S0=1): q = 0.5
- 2-step with stopped model (up→2.0 from 1.5, down→1.05): q_up ≈ 0.4737, q_dn=0.375
- Full model (S2(uu)=2.25): q = 0.375 uniformly (symmetric recombining tree)

## Key implementation notes — exp_1 through exp_4

### exp_1 (FTAP validation)
- Gain matrix A shape (4 paths, 3 basis strategies): rows=paths (uu/ud/du/dd),
  cols=(h_root, h_mid_up, h_mid_dn). Martingale constraint: A.T @ q = 0.
- EMM via cvxpy: minimize 0 subject to q>=eps, sum(q)=1, A.T@q=0.
- Unique EMM: Q=(0.25, 0.25, 0.25, 0.25) for up=1.2, down=0.8, S0=1.
- NA LP: -A@h <= 0 AND -(P@A)@h <= -1. If infeasible (scipy status 2) → NA holds.

### exp_2 (counterexample)
- Arbitrage model A: two scenarios, both ΔS>0 → q·ΔS=0 contradicts q>0. INFEASIBLE.
- Analytical EMM check: q_1 = (S0-S_T^2)/(S_T^1-S_T^2). If <0 or >1, no EMM.
- Model B (d=2): A.T@q=0 with ΔS matrix all positive → infeasible EMM.

### exp_3 (geometry lemma)
- u_J computed via SLSQP: max <u, Σz> s.t. <u,z>≥0 ∀z∈J, |u|²≤1. u_J=0 for balanced J.
- J0 = {z: |<u_J,z>|<tol}; for balanced 2d redundant model, J0=J always.
- β = min_{v∈span(J0),|v|=1} max_{z∈J0}(-<v,z>) via random-restart SLSQP.
- κ = 1/β. For J={(δy,δy),(-δy,-δy)}: β=δy·√2, κ=1/(δy·√2).
- Per-step kappas: κ differs per step because |ΔS| differs (use mid-specific kappa in R_t).
- With correct per-step κ: each step multiplies R by exactly 2.0; R_T=4 for 2-step model.
- Projection onto span(J): G = P_J H where P_J = V V.T (SVD). For S=(Y,Y): G=(h1+h2)/2·(1,1).
- Redundant position H=(100,-100) → G=(0,0): gain=0, trivially bounded.

### exp_4 (L²(μ) compactness)
- μ-weight = R_T^{-3} · c^{-2} · |ΔS| (scalar for this constant model).
- Gram matrix M[i,j] = MU_WEIGHT · Σ_paths P_path · Σ_steps h^i_{step} · h^j_{step}.
- Tail average Ĝ^n = (1/(N-n+1)) · Σ_{j≥n} G^j. Residuals ||Ĝ^n-G^N||_μ → 0.
- For monotone convergence: strategies must converge (e.g., holdings decrease monotonically).
- cvxpy QP: min w^T M w s.t. w≥0, sum(w)=1. Optimal w puts full weight on strategy with min norm.
- Bound: ||G^n||²_{L²(μ)} ≤ E[V_T/R_T] = (N_steps·|ΔS|)/R_T.

## Test counts
- test_exp_1_ftap.py:         9 tests
- test_exp_2_counterexample.py: 6 tests
- test_exp_3_geometry.py:     9 tests
- test_exp_7_local_martingale.py: 8 tests
- Total: 32 tests
