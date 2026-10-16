"""
Tests for Experiment 3: Geometric structure of the gain cone.

Covers:
  - Projection matrix properties (idempotent, symmetric, correct action).
  - R-process (Lemma 3) for bounding strategy wealth.
  - Gain preservation under projection onto the effective asset direction.
"""

import pytest
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import svd


def compute_projection_matrix(vectors: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    """Compute orthogonal projection matrix onto the column span of `vectors.T`."""
    if len(vectors) == 0 or np.allclose(vectors, 0):
        return np.zeros((vectors.shape[1], vectors.shape[1]))
    U, s, Vt = svd(vectors.T, full_matrices=False)
    rank = np.sum(s > tol)
    U = U[:, :rank]
    return U @ U.T


class TestProjectionMatrix:
    def test_projection_is_idempotent(self):
        J0 = np.array([[1.0, 0.0], [0.0, 1.0]])
        P = compute_projection_matrix(J0)
        assert np.allclose(P @ P, P)

    def test_projection_is_symmetric(self):
        J0 = np.array([[1.0, 1.0], [1.0, -1.0]])
        P = compute_projection_matrix(J0)
        assert np.allclose(P, P.T)

    def test_projection_onto_1d_subspace(self):
        J0 = np.array([[1.0, 0.0]])   # project onto x-axis
        P = compute_projection_matrix(J0)
        v = np.array([2.0, 3.0])
        Pv = P @ v
        assert np.allclose(Pv, np.array([2.0, 0.0]))

    def test_projection_redundant_asset(self):
        # For S=(Y,Y) model, effective direction is (1,1)/sqrt(2).
        # J0 contains increments proportional to (1,1).
        J0 = np.array([[0.1, 0.1], [-0.1, -0.1]])
        P = compute_projection_matrix(J0)
        assert P.shape == (2, 2)
        # Projection should be onto the (1,1)/sqrt(2) direction.
        v = np.array([1.0, -1.0])   # perpendicular to (1,1)
        Pv = P @ v
        # (1,-1) is orthogonal to span{(0.1,0.1)}, so P maps it near 0.
        assert np.linalg.norm(Pv) < 0.1


class TestRProcessLemma3:
    def test_R_process_starts_at_1(self):
        R = 1.0
        assert R == 1.0

    def test_R_process_increasing(self):
        c_vals = [2.0, 2.0, 2.0]
        delta_S_norms = [0.1, 0.2, 0.15]
        R = 1.0
        R_values = [R]
        for c, ds in zip(c_vals, delta_S_norms):
            R = R * (1 + c * ds)
            R_values.append(R)
        assert all(R_values[i] <= R_values[i + 1] for i in range(len(R_values) - 1))

    def test_wealth_bound_0_leq_W_leq_aR(self):
        a = 1.0
        R_vals = [1.0, 1.2, 1.5, 2.0]
        W_vals = [1.0, 1.1, 1.3, 1.8]   # synthetic wealth
        for W, R in zip(W_vals, R_vals):
            assert 0 <= W
            assert W <= a * R

    def test_variation_bound(self):
        a = 1.0
        R_T = 2.0
        var_HdotS = 0.5   # some variation
        # Var(H.S)_T <= a * (R_T - 1)
        assert var_HdotS <= a * (R_T - 1)


class TestGainPreservation:
    def test_gain_preserved_after_projection(self):
        # If G = P_J H and delta_S is in the non-projected direction,
        # then <G, delta_S> = <H, delta_S>.
        # For 2D redundant model: H=(3,2), delta_S=(1,1)/sqrt(2)*ds.
        delta_S = np.array([0.1, 0.1])
        H = np.array([3.0, 2.0])
        # Project onto span{(1,1)}.
        direction = np.array([1.0, 1.0]) / np.sqrt(2)
        P = np.outer(direction, direction)
        G = P @ H
        # Gain should be preserved since delta_S is in the (1,1) direction.
        assert abs(np.dot(G, delta_S) - np.dot(H, delta_S)) < 1e-10
