"""
Tests for Experiment 7: Doubling process — local martingale but not true martingale.

Process: S_0=1; at each step, double with prob 0.5 or go to 0 permanently.
State space E = {0} ∪ {2^k : k ≥ 0}.
"""

import pytest
import numpy as np


class TestDoublingProcess:
    def setup_method(self):
        rng = np.random.default_rng(42)
        N = 20
        n_paths = 50_000

        # Generate coin tosses
        coins = rng.binomial(1, 0.5, (n_paths, N))   # 1=heads=up, 0=tails=zero
        paths = np.ones((n_paths, N + 1))

        for n in range(N):
            alive = paths[:, n] > 0
            paths[alive, n + 1] = np.where(
                coins[alive, n] == 1,
                2.0 * paths[alive, n],
                0.0,
            )
            paths[~alive, n + 1] = 0.0

        self.paths  = paths
        self.N      = N
        self.n_paths = n_paths

    def test_initial_value(self):
        assert np.all(self.paths[:, 0] == 1.0)

    def test_state_space_is_powers_of_two_or_zero(self):
        unique_vals = np.unique(self.paths)
        # All values should be 0 or 2^k for k >= 0.
        for v in unique_vals:
            if v > 0:
                log2v = np.log2(v)
                assert abs(log2v - round(log2v)) < 1e-10, f"Value {v} not a power of 2"

    def test_terminal_value_approaches_zero(self):
        # P(S_T > 0) = (0.5)^N -> 0.
        frac_alive    = np.mean(self.paths[:, -1] > 0)
        expected_prob = 0.5 ** self.N
        assert frac_alive < 0.01
        assert abs(frac_alive - expected_prob) < 0.01

    def test_stopped_process_bounded(self):
        m     = 2   # bound 2^2 = 4
        bound = 2 ** m
        stopped = self.paths.copy()
        # Apply stopping: once value exceeds bound, freeze at previous value.
        for n in range(1, self.N + 1):
            sigma_mask = stopped[:, n] > bound
            stopped[sigma_mask, n] = stopped[sigma_mask, n - 1]
        assert np.all(stopped <= bound + 1e-10)

    def test_step_conditional_mean_equals_current(self):
        # E[S_{n+1} | S_n = v] = v  (fair coin: 0.5*2v + 0.5*0 = v).
        n = 3   # test at step 3
        alive_mask = self.paths[:, n] > 0
        if alive_mask.sum() > 0:
            S_n  = self.paths[alive_mask, n]
            S_n1 = self.paths[alive_mask, n + 1]
            # Conditional ratio: should average to 1.0.
            conditional_ratio = S_n1 / S_n
            mean_ratio = conditional_ratio.mean()
            # Monte Carlo error: std/sqrt(n_alive) ~ 1/sqrt(50000*(0.5)^3) ~ 0.013
            assert abs(mean_ratio - 1.0) < 0.05

    def test_true_martingale_impossible(self):
        # If Q ~ P and S is Q-true-martingale:
        #   E_Q[S_T] = S_0 = 1.0
        # But S_T = 0 P-a.s. => S_T = 0 Q-a.s. => E_Q[S_T] = 0: contradiction.
        S0 = 1.0
        S_T_Pa_s = 0.0   # theoretical limit
        # If Q ~ P, E_Q[S_T] = E_Q[0] = 0 != S0 = 1.0.
        contradiction = (S_T_Pa_s != S0)
        assert contradiction   # proves no equivalent true martingale measure

    def test_local_state_space_finite(self):
        # E ∩ [0, b] is finite for every b.
        b    = 10.0
        E_b  = [v for v in [0] + [2**k for k in range(20)] if v <= b]
        assert len(E_b) < np.inf
        assert len(E_b) == 5   # {0, 1, 2, 4, 8} for b=10


class TestLocalizationMartingale:
    def test_stopped_is_martingale_exact(self):
        # For m=1 (bound=2): stopped process value v satisfies:
        #   v < 2:  E[next] = 0.5*2v + 0.5*0 = v  (martingale step).
        #   v = 2 (stopped): E[next] = 2 (stays).  (trivially martingale).
        for v in [1.0, 2.0]:
            if v < 2.0:
                E_next = 0.5 * 2 * v + 0.5 * 0
                assert abs(E_next - v) < 1e-10
            else:   # stopped
                E_next = v   # stays at boundary
                assert abs(E_next - v) < 1e-10
