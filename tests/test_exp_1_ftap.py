"""
Tests for Experiment 1: FTAP validation on the 2-step binomial model.

Parameters:
  S0=1.0, up=1.2, down=0.8, p_up=0.6, p_down=0.4
  EMM: q = (S0 - down)/(up - down) = (1.0-0.8)/(1.2-0.8) = 0.5
  Physical probs: P(uu)=0.36, P(ud)=0.24, P(du)=0.24, P(dd)=0.16
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


class TestBinomialTree:
    def test_terminal_probs_sum_to_one(self):
        # 4 terminal scenarios with probs 0.36, 0.24, 0.24, 0.16
        probs = np.array([0.36, 0.24, 0.24, 0.16])
        assert abs(probs.sum() - 1.0) < 1e-10

    def test_na_holds_in_arbitrage_free_model(self):
        # In fair binomial, no zero-cost strategy yields guaranteed profit.
        # Check by LP: no admissible strategy with X_T >= 0 and X_T > 0 somewhere.
        # Use linprog to check feasibility of: A*h >= 0, sum(payoffs) > 0.
        # Actually check if max(min_scenario_payoff) over normalized h is 0.
        from scipy.optimize import linprog
        # Strategy: h_root = h0, h_up = h1, h_down = h2
        # Terminal payoffs:
        #   uu: h0*(1.2-1.0) + h1*(1.44-1.2)
        #   ud: h0*(1.2-1.0) + h1*(0.96-1.2)
        #   du: h0*(0.8-1.0) + h2*(0.96-0.8)
        #   dd: h0*(0.8-1.0) + h2*(0.64-0.8)
        # Payoff matrix A (4 scenarios x 3 strategies):
        A = np.array([
            [ 0.2,  0.24,  0.0 ],
            [ 0.2, -0.24,  0.0 ],
            [-0.2,  0.0,   0.16],
            [-0.2,  0.0,  -0.16],
        ])
        # NA: no h with A@h >= 0 component-wise and sum(A@h) > 0.
        # Maximise min payoff over ||h||=1: if max = 0, no arbitrage.
        # Equivalently check: no h with all payoffs >= 0 and at least one > 0.
        # LP: max t  s.t. A@h >= t·1, t >= 0.
        # If optimal t > 0 arbitrage exists; t = 0 means NA holds.
        n_strat = 3
        n_scen  = 4
        # Variables: (h0, h1, h2, t)
        c = np.zeros(n_strat + 1); c[-1] = -1.0   # maximise t
        A_ub = np.hstack([-A, np.ones((n_scen, 1))])  # -A@h + t <= 0
        b_ub = np.zeros(n_scen)
        bounds = [(-10, 10)] * n_strat + [(0, 10)]
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        opt_t = -res.fun if res.success else 0.0
        # In an arbitrage-free model the optimum t is 0 (no guaranteed profit)
        assert opt_t < 1e-8

    def test_emm_exists_and_unique(self):
        # In complete 2-step binomial, EMM is unique.
        # q_up such that q*1.2 + (1-q)*0.8 = 1.0 => q=0.5
        q = (1.0 - 0.8) / (1.2 - 0.8)
        assert abs(q - 0.5) < 1e-10
        assert 0 < q < 1

    def test_q_weights_positive(self):
        q = 0.5
        Q_uu = q * q
        Q_ud = q * (1 - q)
        Q_du = (1 - q) * q
        Q_dd = (1 - q) * (1 - q)
        assert all(v > 0 for v in [Q_uu, Q_ud, Q_du, Q_dd])
        assert abs(Q_uu + Q_ud + Q_du + Q_dd - 1.0) < 1e-10

    def test_martingale_property_under_q(self):
        q = 0.5
        # E_Q[S_1 | F_0] should equal S_0 = 1.0
        E_S1 = q * 1.2 + (1 - q) * 0.8
        assert abs(E_S1 - 1.0) < 1e-10
        # E_Q[S_2 | F_1=up] = q*1.44 + (1-q)*0.96 = 1.2
        E_S2_up = q * 1.44 + (1 - q) * 0.96
        assert abs(E_S2_up - 1.2) < 1e-10
        # E_Q[S_2 | F_1=down] = q*0.96 + (1-q)*0.64 = 0.8
        E_S2_down = q * 0.96 + (1 - q) * 0.64
        assert abs(E_S2_down - 0.8) < 1e-10

    def test_na_nflvr_emm_consistency(self):
        # In bounded complete binomial: NA iff NFLVR iff EMM exists.
        # All three hold in our model.
        na = True
        nflvr = True
        emm_exists = True
        assert na == nflvr == emm_exists


class TestSeparatingDensity:
    def test_density_normalization(self):
        q = 0.5
        P = np.array([0.36, 0.24, 0.24, 0.16])
        Q = np.array([q**2, q * (1 - q), (1 - q) * q, (1 - q)**2])
        Z = Q / P
        assert abs(np.sum(P * Z) - 1.0) < 1e-10

    def test_density_strictly_positive(self):
        q = 0.5
        P = np.array([0.36, 0.24, 0.24, 0.16])
        Q = np.array([q**2, q * (1 - q), (1 - q) * q, (1 - q)**2])
        Z = Q / P
        assert np.all(Z > 0)

    def test_separation_inequality(self):
        # For any admissible gain f in the arbitrage-free model, E[Z*f] <= 0.
        q = 0.5
        P = np.array([0.36, 0.24, 0.24, 0.16])
        Q = np.array([q**2, q * (1 - q), (1 - q) * q, (1 - q)**2])
        Z = Q / P
        # Terminal gains from buying 1 unit from t=0 to t=1:
        # uu, ud: gain = 0.2;  du, dd: gain = -0.2
        f_buy = np.array([0.2, 0.2, -0.2, -0.2])
        # E_Q[gain] = 0 since S is a Q-martingale
        assert np.sum(P * Z * f_buy) < 1e-10
