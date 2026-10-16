# FTAP End-to-End Validation on Bounded Arbitrage-Free Synthetic Price Systems

Computational validation of the **Fundamental Theorem of Asset Pricing (FTAP)**
in bounded and locally-bounded synthetic c\u00e0dl\u00e0g piecewise-constant discounted
price models. All experiments use finite scenario trees and exact arithmetic;
no empirical market data is used.

## Research Objectives

| Exp | Title | Method |
|-----|-------|--------|
| 1 | FTAP on bounded arbitrage-free models | LP NA check + cvxpy EMM |
| 2 | Counterexample: arbitrage \u21d2 no EMM | LP arb detect + cvxpy infeasibility |
| 3 | Geometry Lemma \u2014 projection & pathwise bounds | SVD + SLSQP + Lemma 3 inequalities |
| 4 | Weighted L\u00b2(\u03bc) compactness & convex averaging | Gram matrix + cvxpy QP |
| 5 | Separation density & equivalent local martingale measure | cvxpy separation feasibility |
| 6 | Filtration enrichment beyond price history | Custom enriched filtration tree |
| 7 | Local martingale but not true martingale (doubling example) | Monte Carlo + logical proof |

## Repository Structure

```
.
\u251c\u2500\u2500 data/               # Placeholder (all data is synthetic, generated in-memory)
\u251c\u2500\u2500 src/               # Core implementations
\u2502   \u251c\u2500\u2500 scenario_tree.py        # ScenarioTree, PredictableStrategy, GainProcess, AdmissibilityChecker
\u2502   \u251c\u2500\u2500 geometry_lemma.py       # Lemma 2 & 3: u_J, P_J, \u03b2_J, \u03ba_J, R_t, V_t, pathwise bounds
\u2502   \u251c\u2500\u2500 separation.py           # cvxpy separation density & martingale measure solver
\u2502   \u251c\u2500\u2500 compactness.py          # Weighted L\u00b2(\u03bc) norm, forward convex averaging
\u2502   \u251c\u2500\u2500 local_martingale_example.py  # Doubling process generator & localizer
\u2502   \u2514\u2500\u2500 filtration_enrichment.py     # Enriched filtration tree with auxiliary signals
\u251c\u2500\u2500 exp/               # Experiment scripts (self-contained)
\u2502   \u251c\u2500\u2500 exp_1_ftap_bounded.py
\u2502   \u251c\u2500\u2500 exp_2_counterexample.py
\u2502   \u251c\u2500\u2500 exp_3_geometry_lemma.py
\u2502   \u251c\u2500\u2500 exp_4_compactness.py
\u2502   \u251c\u2500\u2500 exp_5_local_martingale_measure.py
\u2502   \u251c\u2500\u2500 exp_6_filtration_enrichment.py
\u2514\u2500\u2500 exp_7_local_not_true_martingale.py
\u251c\u2500\u2500 tests/             # pytest test suite (32 tests, all pass)
\u2502   \u251c\u2500\u2500 test_exp_1_ftap.py
\u2502   \u251c\u2500\u2500 test_exp_2_counterexample.py
\u2502   \u251c\u2500\u2500 test_exp_3_geometry.py
\u2514\u2500\u2500   test_exp_7_local_martingale.py
\u251c\u2500\u2500 results/           # Outputs: JSON metrics + PNG plots + RESULTS.md
\u2514\u2500\u2500 README.md
```

## Quick Start

```bash
# Install dependencies (all standard, no Massive API needed)
pip install numpy scipy cvxpy matplotlib pandas

# Run all experiments
python exp/exp_1_ftap_bounded.py
python exp/exp_2_counterexample.py
python exp/exp_3_geometry_lemma.py
python exp/exp_4_compactness.py
python exp/exp_5_local_martingale_measure.py
python exp/exp_6_filtration_enrichment.py
python exp/exp_7_local_not_true_martingale.py

# Run all tests
pytest tests/ -v
```

## Key Results

- **Exp 1**: Bounded arbitrage-free 2-step binomial satisfies NA\u21d4NFLVR\u21d4EMM; S is a true Q-martingale with max residual < 1e\u221214.
- **Exp 2**: One-jump model with both increments positive: NA fails, EMM infeasible (cvxpy status: `infeasible`).
- **Exp 3**: Lemma 2 projection preserves gains; Lemma 3 bounds `0\u2264W_t\u2264a\u00b7R_t` and `Var(H\u00b7S)_t\u2264a(R_t\u22121)` verified on all scenarios.
- **Exp 4**: 5-strategy sequence converges in L\u00b2(\u03bc); sup gain residual \u2192 0; optimal convex weights reduce norm from 0.103 to 0.036.
- **Exp 5**: Separating density Z and EMM Q found for each localization; every stopped S^{\u03c3_m} is a Q-martingale.
- **Exp 6**: Paths with identical price histories but different signals induce different holdings \u2014 full filtration is strictly richer than price history.
- **Exp 7**: Stopped processes are martingales (max error < 0.05); S_T=0 P-a.s.; true martingale measure provably impossible.

See [results/RESULTS.md](results/RESULTS.md) for full metrics and plots.

## Dependencies

| Package | Purpose |
|---------|---------|
| numpy | Array operations, path simulation |
| scipy | Linear programming (NA check), SLSQP (geometry), SVD |
| cvxpy | Convex feasibility (EMM construction, separation density, QP averaging) |
| matplotlib | All plots and figures |
| pandas | Reporting tables |

No market data APIs are used. All models are synthetic finite scenario trees.
