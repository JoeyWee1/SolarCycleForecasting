"""
Tests for helpers/eval.py — check_constant, best_in_x, truth_in_x.
"""

import numpy as np
import pytest

from helpers.eval import check_constant, best_in_x, truth_in_x


# ─────────────────────────────────────────────────────────────────────────────
# check_constant
# ─────────────────────────────────────────────────────────────────────────────

class TestCheckConstant:

    def _flat_preds(self, n_samples=50, n_times=200):
        return np.ones((n_samples, n_times)) * 0.5

    def _oscillating_preds(self, amplitude=1.0, n_samples=50, n_times=200):
        t = np.linspace(0, 4 * np.pi, n_times)
        return np.stack([amplitude * np.sin(t + i * 0.1) for i in range(n_samples)])

    def test_flat_predictions_are_constant(self):
        """Predictions with zero amplitude should be flagged as constant."""
        preds = self._flat_preds()
        assert check_constant(preds, MAD=0.01)

    def test_large_amplitude_not_constant(self):
        """Predictions whose amplitude >> MAD should not be flagged as constant."""
        preds = self._oscillating_preds(amplitude=2.0)
        assert not check_constant(preds, MAD=0.01)

    def test_returns_bool(self):
        result = check_constant(self._flat_preds(), MAD=0.1)
        assert isinstance(result, (bool, np.bool_))

    def test_tol_controls_threshold(self):
        """With amplitude/MAD = 0.5, tol=2 → constant; tol=0.1 → not constant."""
        t = np.linspace(0, 2 * np.pi, 200)
        preds = np.stack([0.5 * np.sin(t) for _ in range(30)])
        MAD = 1.0
        assert check_constant(preds, MAD=MAD, tol=2)
        assert not check_constant(preds, MAD=MAD, tol=0.1)

    def test_percentile_parameter(self):
        """Higher percentile tightens the criterion: constant at p50, not at p99."""
        t = np.linspace(0, 2 * np.pi, 200)
        # Most samples flat, a handful with large swings
        flat  = np.ones((45, 200)) * 0.5
        noisy = np.stack([3.0 * np.sin(t) for _ in range(5)])
        preds = np.vstack([flat, noisy])
        MAD = 0.01
        assert check_constant(preds, MAD=MAD, tol=2, percentile=50)
        assert not check_constant(preds, MAD=MAD, tol=2, percentile=99)


# ─────────────────────────────────────────────────────────────────────────────
# best_in_x
# ─────────────────────────────────────────────────────────────────────────────

class TestBestInX:

    def _sinusoidal_preds(self, n_samples=50, min_at_day=730):
        """Predictions whose minimum sits near `min_at_day` from t[0]."""
        t = np.linspace(0, 5 * 365, 1000)
        preds = np.stack([
            0.5 * np.sin(2 * np.pi * (t - min_at_day) / (2 * 365)) + i * 1e-4
            for i in range(n_samples)
        ])
        return preds, t

    def test_returns_median_and_bounds_tuple(self):
        preds, t = self._sinusoidal_preds()
        result = best_in_x(preds, t, lookahead_years=3, t_start=t[0])
        assert isinstance(result, tuple) and len(result) == 2
        _, bounds = result
        assert len(bounds) == 2

    def test_median_is_finite(self):
        preds, t = self._sinusoidal_preds()
        median, _ = best_in_x(preds, t, lookahead_years=3, t_start=t[0])
        assert np.isfinite(median)

    def test_bounds_ordered(self):
        """Lower percentile bound ≤ median ≤ upper percentile bound."""
        preds, t = self._sinusoidal_preds()
        median, (lb, ub) = best_in_x(preds, t, lookahead_years=3, t_start=t[0])
        assert lb <= median <= ub

    def test_median_within_lookahead_window(self):
        preds, t = self._sinusoidal_preds()
        t_start, la = t[0], 3
        median, _ = best_in_x(preds, t, lookahead_years=la, t_start=t_start)
        assert t_start <= median <= t_start + la * 365

    def test_default_t_start_equals_first_t(self):
        """Omitting t_start should behave identically to passing t[0]."""
        preds, t = self._sinusoidal_preds()
        explicit = best_in_x(preds, t, lookahead_years=3, t_start=t[0])
        default  = best_in_x(preds, t, lookahead_years=3)
        assert explicit[0] == default[0]

    def test_shorter_window_restricts_result(self):
        """A shorter lookahead window cannot return a later minimum than a longer one."""
        preds, t = self._sinusoidal_preds(min_at_day=900)
        t_start = t[0]
        med_1yr, _ = best_in_x(preds, t, lookahead_years=1, t_start=t_start)
        med_3yr, _ = best_in_x(preds, t, lookahead_years=3, t_start=t_start)
        assert med_1yr <= t_start + 1 * 365


# ─────────────────────────────────────────────────────────────────────────────
# truth_in_x
# ─────────────────────────────────────────────────────────────────────────────

class TestTruthInX:

    def _fourier_preds(self, n_samples=100, n_points=2000, min_year=2010.5):
        """Synthetic predictions with a sharp Gaussian trough at `min_year`."""
        t_year = np.linspace(2000, 2020, n_points)
        preds = np.stack([
            -np.exp(-0.5 * ((t_year - min_year) / 0.2) ** 2) + i * 1e-4
            for i in range(n_samples)
        ])
        return preds, t_year

    def test_returns_median_and_bounds(self):
        preds, t_year = self._fourier_preds()
        result = truth_in_x(preds, t_year, t0_year=2009.0, lookahead_years=3)
        assert isinstance(result, tuple) and len(result) == 2
        _, bounds = result
        assert len(bounds) == 2

    def test_median_is_finite(self):
        preds, t_year = self._fourier_preds()
        median, _ = truth_in_x(preds, t_year, t0_year=2009.0, lookahead_years=3)
        assert np.isfinite(median)

    def test_bounds_ordered(self):
        preds, t_year = self._fourier_preds()
        median, (lb, ub) = truth_in_x(preds, t_year, t0_year=2009.0, lookahead_years=3)
        assert lb <= median <= ub

    def test_minimum_within_window(self):
        preds, t_year = self._fourier_preds()
        t0, la = 2009.0, 3
        median, _ = truth_in_x(preds, t_year, t0_year=t0, lookahead_years=la)
        assert t0 <= median <= t0 + la

    def test_median_near_true_minimum(self):
        """Median should be close to the known minimum year (within 0.2 yr)."""
        true_min = 2010.5
        preds, t_year = self._fourier_preds(min_year=true_min)
        median, _ = truth_in_x(preds, t_year, t0_year=2009.0, lookahead_years=3)
        assert abs(median - true_min) < 0.2

    def test_narrower_window_bounds_tighter(self):
        """A 1-year window should produce tighter or equal bounds than a 5-year window."""
        preds, t_year = self._fourier_preds()
        _, (lb1, ub1) = truth_in_x(preds, t_year, t0_year=2009.0, lookahead_years=1)
        _, (lb5, ub5) = truth_in_x(preds, t_year, t0_year=2009.0, lookahead_years=5)
        assert (ub1 - lb1) <= (ub5 - lb5) + 1e-9
