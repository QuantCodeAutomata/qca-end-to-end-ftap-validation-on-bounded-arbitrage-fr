"""
Tests for Experiment 2: Arbitrage counterexample.

Model: one-period, d=1, two scenarios.
  S0 = 1.0
  S1 outcomes: 1.2 (prob 0.6) and 1.1 (prob 0.4)
  delta_S = (0.2, 0.1) — both positive.
  Any H > 0 yields strictly positive gains in both scenarios → NA fails.
  No equivalent martingale measure exists (S0 < min(S1)).
"""

import pytest
import numpy as np


class TestArbitrageModel:
    def test_arbitrage_strategy_exists(self):
        # Both jumps positive: delta_S = (0.2, 0.1)
        delta_S = np.array([0.2, 0.1])
        h = 1.0
        gains = h * delta_S
        assert np.all(gains > 0)

    def test_na_fails(self):
        # X_T = h*delta_S, all > 0, so NA fails.
        delta_S = np.array([0.2, 0.1])
        h = 1.0
        X_T = h * delta_S
        na_satisfied = not (np.all(X_T >= 0) and np.any(X_T > 0))
        assert not na_satisfied   # NA should fail

    def test_no_separating_density_feasible(self):
        # Separation requires 0.6*z1*0.2 + 0.4*z2*0.1 <= 0 with z1,z2>0: impossible.
        # Verify mathematically.
        P = np.array([0.6, 0.4])
        f = np.array([0.2, 0.1])   # positive gains
        # For any z1, z2 > 0: sum(P_k * z_k * f_k) > 0 (all terms positive).
        for z1, z2 in [(1, 1), (2, 0.5), (0.1, 10)]:
            sep_val = P[0] * z1 * f[0] + P[1] * z2 * f[1]
            assert sep_val > 0   # cannot be <= 0

    def test_no_martingale_measure(self):
        # martingale: q1*1.2 + q2*1.1 = 1.0 with q1+q2=1, q1,q2>0
        # => 1.1 + 0.1*q1 = 1.0 => q1 = -1 < 0: impossible.
        # Equivalently, S_0=1 < min(S_1 outcomes)=1.1, so no EMM.
        S0 = 1.0
        outcomes = np.array([1.2, 1.1])
        assert S0 < outcomes.min()   # S0 below all outcomes: no martingale measure


class TestNAImpliesEMMInArbitrageFreeModel:
    """Consistency check: in arbitrage-free models, EMM always exists."""

    def test_fair_binomial_has_emm(self):
        S0 = 1.0
        up = 1.2
        down = 0.8
        q = (S0 - down) / (up - down)
        assert 0 < q < 1   # valid probability

    def test_emm_probability_in_01(self):
        S0 = 1.0
        up = 1.2
        down = 0.8
        q = (S0 - down) / (up - down)
        assert 0 < q < 1
        assert abs(q * up + (1 - q) * down - S0) < 1e-10
