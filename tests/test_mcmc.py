"""
Tests for helpers/MCMC.py — lnPost_gp.
"""

import numpy as np
import pytest
import celerite2
from celerite2 import terms

from helpers.gpr import set_params
from helpers.MCMC import lnPost_gp


# ─────────────────────────────────────────────────────────────────────────────
# Shared fixture
# ─────────────────────────────────────────────────────────────────────────────

def _make_gp(k=1, n=60, seed=42):
    """
    Return (gp, y, log_params, wide_bounds) for a k-term SHO kernel.
    The GP has already been computed on a synthetic dataset.
    """
    np.random.seed(seed)
    t = np.sort(np.random.uniform(0, 1000, n))
    y = 0.2 + 0.01 * np.sin(2 * np.pi * t / 365)

    sigma_0, rho_0, q_0 = 0.01, 365.0, 2.0
    kernel = terms.SHOTerm(sigma=sigma_0, rho=rho_0, Q=q_0)
    for _ in range(1, k):
        kernel += terms.SHOTerm(sigma=sigma_0, rho=rho_0 * 2, Q=q_0)

    gp = celerite2.GaussianProcess(kernel, mean=float(np.mean(y)))
    gp.compute(t, yerr=y * 0.025)

    log_params = np.array(
        [np.log(sigma_0)] * k + [np.log(rho_0)] * k + [np.log(q_0)] * k
    )
    wide_bounds = np.column_stack([log_params - 10, log_params + 10])
    return gp, y, log_params, wide_bounds


# ─────────────────────────────────────────────────────────────────────────────
# lnPost_gp — unit tests
# ─────────────────────────────────────────────────────────────────────────────

class TestLnPostGP:

    def test_returns_finite_for_valid_params(self):
        """At sensible parameters inside wide bounds, the posterior must be finite."""
        gp, y, log_params, bounds = _make_gp(k=1)
        result = lnPost_gp(log_params, gp, k=1, y=y,
                           initial_guesses=log_params, bounds=bounds)
        assert np.isfinite(result)

    def test_returns_neg_inf_below_lower_bound(self):
        """Parameters below the lower bound must return -inf."""
        gp, y, log_params, _ = _make_gp(k=1)
        tight_bounds = np.column_stack([log_params + 5, log_params + 10])
        result = lnPost_gp(log_params, gp, k=1, y=y,
                           initial_guesses=log_params, bounds=tight_bounds)
        assert result == -np.inf

    def test_returns_neg_inf_above_upper_bound(self):
        """Parameters above the upper bound must return -inf."""
        gp, y, log_params, _ = _make_gp(k=1)
        tight_bounds = np.column_stack([log_params - 10, log_params - 5])
        result = lnPost_gp(log_params, gp, k=1, y=y,
                           initial_guesses=log_params, bounds=tight_bounds)
        assert result == -np.inf

    def test_returns_float(self):
        gp, y, log_params, bounds = _make_gp(k=1)
        result = lnPost_gp(log_params, gp, k=1, y=y,
                           initial_guesses=log_params, bounds=bounds)
        assert isinstance(result, float)

    def test_prior_penalty_lowers_posterior_far_from_mode(self):
        """Params far from initial_guesses should score lower with a tight prior."""
        gp, y, log_params, bounds = _make_gp(k=1)
        at_mode  = lnPost_gp(log_params,       gp, 1, y, log_params, bounds, prior_std=1.0)
        far_away = lnPost_gp(log_params + 3.0, gp, 1, y, log_params, bounds, prior_std=1.0)
        assert at_mode > far_away

    def test_wide_prior_reduces_penalty(self):
        """A very wide prior_std should penalise distant params much less than a tight one."""
        gp, y, log_params, bounds = _make_gp(k=1)
        perturbed = log_params + 2.0
        tight  = lnPost_gp(perturbed, gp, 1, y, log_params, bounds, prior_std=0.5)
        loose  = lnPost_gp(perturbed, gp, 1, y, log_params, bounds, prior_std=100.0)
        assert loose > tight

    def test_two_term_kernel(self):
        """lnPost_gp must handle k=2 without error."""
        gp, y, log_params, bounds = _make_gp(k=2)
        result = lnPost_gp(log_params, gp, k=2, y=y,
                           initial_guesses=log_params, bounds=bounds)
        assert np.isfinite(result)

    def test_three_term_kernel(self):
        """lnPost_gp must handle k=3 without error."""
        gp, y, log_params, bounds = _make_gp(k=3)
        result = lnPost_gp(log_params, gp, k=3, y=y,
                           initial_guesses=log_params, bounds=bounds)
        assert np.isfinite(result)

    def test_bounds_shape_mismatch_handled(self):
        """Bounds with one param outside range should still return -inf, not raise."""
        gp, y, log_params, bounds = _make_gp(k=1)
        # Make the first param violate its lower bound
        bad_bounds = bounds.copy()
        bad_bounds[0, 0] = log_params[0] + 1.0  # lower bound above current param
        result = lnPost_gp(log_params, gp, k=1, y=y,
                           initial_guesses=log_params, bounds=bad_bounds)
        assert result == -np.inf
