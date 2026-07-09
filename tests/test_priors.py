"""
Tests for the prior generating pipeline (mirrors the analysis in priors5).

Unit tests verify the SM2016 math and get_priors resolver logic in isolation.
Integration tests run find_classify_signals + get_priors on the three Mt. Wilson
benchmark stars and assert the detected periods are within 20% of published
literature values.

Literature values
-----------------
HD81809  (G): P_rot = 39.3 d,  P_cyc = 8.2 yr   (Baliunas et al. 1995)
HD160346 (K): P_rot = 35.3 d,  P_cyc = 7.0 yr
HD201091 (K): P_rot = 34.0 d,  P_cyc = 7.3 yr
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from helpers.priors import (
    SM2016_intercepts,
    SM2016_s_to_m,
    SM2016_m_to_s,
    find_classify_signals,
    get_priors,
)
from helpers.df_ops import prepare_df, split_df

DATA_DIR = Path(__file__).parents[1] / "Data" / "benchmark"

# Benchmark star metadata: data file, spectral type, literature periods
BENCHMARK_STARS = {
    "HD81809": {
        "path":           DATA_DIR / "HD81809_Mt_wilson_data.txt",
        "star_type":      "G",
        "lit_short_days": 39.3,
        "lit_mid_days":   8.2 * 365,
    },
    "HD160346": {
        "path":           DATA_DIR / "HD160346_Mt_wilson_data.txt",
        "star_type":      "K",
        "lit_short_days": 35.3,
        "lit_mid_days":   7.0 * 365,
    },
    "HD201091": {
        "path":           DATA_DIR / "HD201091_Mt_wilson_data.txt",
        "star_type":      "K",
        "lit_short_days": 34.0,
        "lit_mid_days":   7.3 * 365,
    },
}

# ── SM2016 mean values (Table 1 & 2 from Selião & Mesquita 2016) ─────────────
SM2016_MEANS = {
    "F": {"P_rot": 8.6,  "P_cyc_yr": 9.5},
    "G": {"P_rot": 19.6, "P_cyc_yr": 6.7},
    "K": {"P_rot": 27.4, "P_cyc_yr": 8.5},
}


# ─────────────────────────────────────────────────────────────────────────────
# SM2016 relation — unit tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSM2016Relations:
    """Unit tests for the SM2016 rotation–cycle period scaling relations."""

    def test_s_to_m_g_star_mean_roundtrip(self):
        """At the G-star mean rotation, SM2016 should return the G-star mean cycle."""
        p_rot = SM2016_MEANS["G"]["P_rot"]
        p_cyc_expected = SM2016_MEANS["G"]["P_cyc_yr"] * 365
        p_cyc = SM2016_s_to_m(p_rot, "G")
        rel_err = abs(p_cyc - p_cyc_expected) / p_cyc_expected
        assert rel_err < 0.01, (
            f"G-star s_to_m at mean P_rot: expected {p_cyc_expected:.0f} d, got {p_cyc:.0f} d"
        )

    def test_m_to_s_k_star_mean_roundtrip(self):
        """At the K-star mean cycle, SM2016 should return the K-star mean rotation."""
        p_cyc = SM2016_MEANS["K"]["P_cyc_yr"] * 365
        p_rot_expected = SM2016_MEANS["K"]["P_rot"]
        p_rot = SM2016_m_to_s(p_cyc, "K")
        rel_err = abs(p_rot - p_rot_expected) / p_rot_expected
        assert rel_err < 0.01, (
            f"K-star m_to_s at mean P_cyc: expected {p_rot_expected} d, got {p_rot:.2f} d"
        )

    @pytest.mark.parametrize("star_type", ["F", "G", "K"])
    def test_round_trip_consistency(self, star_type):
        """s_to_m followed by m_to_s should recover the original rotation period."""
        p_rot_in = SM2016_MEANS[star_type]["P_rot"]
        p_cyc = SM2016_s_to_m(p_rot_in, star_type)
        p_rot_out = SM2016_m_to_s(p_cyc, star_type)
        assert abs(p_rot_out - p_rot_in) < 0.01, (
            f"{star_type}: round trip {p_rot_in:.1f} → {p_cyc:.1f} → {p_rot_out:.3f} (should be {p_rot_in})"
        )

    def test_intercepts_returns_fgk_keys(self):
        """SM2016_intercepts must return a dict with F, G, K keys."""
        c = SM2016_intercepts(0.89)
        assert set(c.keys()) >= {"F", "G", "K"}

    def test_longer_rotation_gives_longer_cycle(self):
        """Longer rotation period should map to a longer cycle period (monotonic)."""
        p_cyc_25 = SM2016_s_to_m(25, "K")
        p_cyc_35 = SM2016_s_to_m(35, "K")
        assert p_cyc_35 > p_cyc_25

    def test_output_is_positive(self):
        """All SM2016 period outputs must be positive."""
        assert SM2016_s_to_m(20, "G") > 0
        assert SM2016_m_to_s(2000, "G") > 0


# ─────────────────────────────────────────────────────────────────────────────
# get_priors resolver — unit tests
# get_priors returns (rho_priors, rho_bounds, sources)
# ─────────────────────────────────────────────────────────────────────────────

class TestGetPriorsResolver:
    """Unit tests for the fallback logic inside get_priors."""

    def test_all_detected_returns_unchanged(self):
        """When all three signals are detected, get_priors passes them straight through."""
        classified = {"short": 35.0, "mid": 2500.0, "long": 12000.0}
        rho_priors, _, _ = get_priors(classified, "K", verbose=False)
        assert rho_priors["short"] == 35.0
        assert rho_priors["mid"]   == 2500.0
        assert rho_priors["long"]  == 12000.0

    def test_missing_mid_derived_from_short_via_sm2016(self):
        """When mid is absent but short is present, mid is derived via SM2016."""
        p_rot = SM2016_MEANS["K"]["P_rot"]
        classified = {"short": p_rot}
        rho_priors, _, sources = get_priors(classified, "K", verbose=False)
        expected_mid = SM2016_s_to_m(p_rot, "K")
        assert abs(rho_priors["mid"] - expected_mid) < 0.01
        assert sources["mid"] == "SM2016"

    def test_missing_short_derived_from_mid_via_sm2016(self):
        """When short is absent but mid is present, short is derived via SM2016."""
        p_cyc = SM2016_MEANS["K"]["P_cyc_yr"] * 365
        classified = {"mid": p_cyc}
        rho_priors, _, sources = get_priors(classified, "K", verbose=False)
        expected_short = SM2016_m_to_s(p_cyc, "K")
        assert abs(rho_priors["short"] - expected_short) < 0.01
        assert sources["short"] == "SM2016"

    @pytest.mark.parametrize("star_type,expected_short,expected_mid_yr", [
        ("F", 8.6,  9.5),
        ("G", 19.6, 6.7),
        ("K", 27.4, 8.5),
    ])
    def test_nothing_detected_uses_sm2016_means(self, star_type, expected_short, expected_mid_yr):
        """When nothing is detected, fall back to star-type mean rotation and cycle periods."""
        rho_priors, _, sources = get_priors({}, star_type, verbose=False)
        assert abs(rho_priors["short"] - expected_short) < 0.01, (
            f"{star_type}: short expected {expected_short}, got {rho_priors['short']:.2f}"
        )
        assert abs(rho_priors["mid"] - expected_mid_yr * 365) < 1.0, (
            f"{star_type}: mid expected {expected_mid_yr * 365:.0f} d, got {rho_priors['mid']:.0f} d"
        )
        assert sources["short"] == "mean"
        assert sources["mid"]   == "mean"

    def test_missing_long_defaults_to_100_years(self):
        """Undetected long period defaults to 100 years (36 500 days)."""
        rho_priors, _, _ = get_priors({"short": 35.0, "mid": 2500.0}, "K", verbose=False)
        assert abs(rho_priors["long"] - 100 * 365) < 1.0

    def test_all_priors_positive(self):
        """Every prior returned by get_priors must be a positive number."""
        rho_priors, _, _ = get_priors({}, "G", verbose=False)
        assert all(v > 0 for v in rho_priors.values()), f"Non-positive prior: {rho_priors}"

    def test_returns_all_three_keys(self):
        """get_priors must always return 'short', 'mid', and 'long' in rho_priors."""
        rho_priors, _, _ = get_priors({}, "G", verbose=False)
        assert set(rho_priors.keys()) == {"short", "mid", "long"}

    def test_returns_three_tuple(self):
        """get_priors must return a 3-tuple of (priors, bounds, sources)."""
        result = get_priors({}, "G", verbose=False)
        assert len(result) == 3

    def test_bounds_contain_prior_value(self):
        """Each prior value must lie within its own bounds."""
        classified = {"short": 35.0, "mid": 2500.0, "long": 12000.0}
        rho_priors, rho_bounds, _ = get_priors(classified, "K", verbose=False)
        for key in ("short", "mid", "long"):
            lb, ub = rho_bounds[key]
            assert lb <= rho_priors[key] <= ub, (
                f"{key}: prior {rho_priors[key]:.1f} not in bounds [{lb:.1f}, {ub:.1f}]"
            )

    def test_bounds_lower_less_than_upper(self):
        """Lower bound must be strictly less than upper bound for each period."""
        rho_priors, rho_bounds, _ = get_priors({}, "G", verbose=False)
        for key in ("short", "mid", "long"):
            lb, ub = rho_bounds[key]
            assert lb < ub, f"{key}: lower bound {lb:.1f} >= upper bound {ub:.1f}"

    def test_direct_detection_source_is_found(self):
        """Directly detected periods must report source='found'."""
        classified = {"short": 35.0, "mid": 2500.0, "long": 12000.0}
        _, _, sources = get_priors(classified, "K", verbose=False)
        assert sources["short"] == "found"
        assert sources["mid"]   == "found"
        assert sources["long"]  == "found"

    def test_sources_keys_match_priors_keys(self):
        """sources dict must have the same keys as rho_priors."""
        rho_priors, _, sources = get_priors({}, "G", verbose=False)
        assert set(sources.keys()) == set(rho_priors.keys())


# ─────────────────────────────────────────────────────────────────────────────
# Integration tests — benchmark stars
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module", params=list(BENCHMARK_STARS.keys()))
def benchmark_priors(request):
    """
    Compute find_classify_signals + get_priors on a benchmark star's training split.
    Scoped to module so each star is processed once regardless of how many tests use it.
    """
    star_name = request.param
    info = BENCHMARK_STARS[star_name]

    raw_df = pd.read_csv(str(info["path"]), sep=r"\s+", skip_blank_lines=True)
    data_df = prepare_df(raw_df)
    train_df, _, _ = split_df(data_df)

    classified = find_classify_signals(
        train_df,
        plot_fitpeaks=False, verbose_fitpeaks=False,
        plot_genpriors=False, verbose_genpriors=False,
    )
    rho_priors, rho_bounds, sources = get_priors(classified, info["star_type"], verbose=False)
    return star_name, info, classified, rho_priors, rho_bounds, sources


_SHORT_TOL = 0.20  # 20% relative tolerance on rotation period
_MID_TOL   = 0.20  # 20% relative tolerance on cycle period


@pytest.mark.slow
def test_short_period_detected(benchmark_priors):
    """find_classify_signals must detect a short (rotation) period for each benchmark star."""
    star_name, _, classified, _, _, _ = benchmark_priors
    assert "short" in classified, (
        f"{star_name}: no short-range signal detected — LSP may have missed the rotation period"
    )


@pytest.mark.slow
def test_mid_period_detected(benchmark_priors):
    """find_classify_signals must detect a mid (cycle) period for each benchmark star."""
    star_name, _, classified, _, _, _ = benchmark_priors
    assert "mid" in classified, (
        f"{star_name}: no mid-range signal detected — LSP may have missed the activity cycle"
    )


@pytest.mark.slow
def test_short_prior_near_literature(benchmark_priors):
    """The detected short prior must be within 20% of the published rotation period."""
    star_name, info, _, rho_priors, _, _ = benchmark_priors
    lit = info["lit_short_days"]
    got = rho_priors["short"]
    rel_err = abs(got - lit) / lit
    assert rel_err < _SHORT_TOL, (
        f"{star_name}: short prior = {got:.2f} d, literature = {lit} d "
        f"(relative error {rel_err * 100:.1f}% > {_SHORT_TOL * 100:.0f}%)"
    )


@pytest.mark.slow
def test_mid_prior_near_literature(benchmark_priors):
    """The detected mid prior must be within 20% of the published activity cycle period."""
    star_name, info, _, rho_priors, _, _ = benchmark_priors
    lit = info["lit_mid_days"]
    got = rho_priors["mid"]
    rel_err = abs(got - lit) / lit
    assert rel_err < _MID_TOL, (
        f"{star_name}: mid prior = {got:.0f} d ({got / 365:.2f} yr), "
        f"literature = {lit:.0f} d ({lit / 365:.2f} yr) "
        f"(relative error {rel_err * 100:.1f}% > {_MID_TOL * 100:.0f}%)"
    )


@pytest.mark.slow
def test_short_prior_in_short_range(benchmark_priors):
    """Short prior must sit within the short-range window (≤ 200 d)."""
    star_name, _, _, rho_priors, _, _ = benchmark_priors
    assert rho_priors["short"] <= 200, (
        f"{star_name}: short prior {rho_priors['short']:.1f} d exceeds the 200 d threshold"
    )


@pytest.mark.slow
def test_mid_prior_in_mid_range(benchmark_priors):
    """Mid prior must sit within the mid-range window (200 d < P ≤ 7 300 d)."""
    star_name, _, _, rho_priors, _, _ = benchmark_priors
    assert 200 < rho_priors["mid"] <= 20 * 365, (
        f"{star_name}: mid prior {rho_priors['mid']:.0f} d is outside (200, 7300] d"
    )


@pytest.mark.slow
def test_long_prior_is_long(benchmark_priors):
    """Long prior must exceed the mid-range upper limit (> 7 300 d)."""
    star_name, _, _, rho_priors, _, _ = benchmark_priors
    assert rho_priors["long"] > 20 * 365, (
        f"{star_name}: long prior {rho_priors['long']:.0f} d is not longer than 7300 d"
    )


@pytest.mark.slow
def test_benchmark_bounds_contain_prior(benchmark_priors):
    """For benchmark stars, each prior must lie within its own detected bounds."""
    star_name, _, _, rho_priors, rho_bounds, _ = benchmark_priors
    for key in ("short", "mid", "long"):
        lb, ub = rho_bounds[key]
        assert lb <= rho_priors[key] <= ub, (
            f"{star_name} {key}: prior {rho_priors[key]:.1f} not in bounds [{lb:.1f}, {ub:.1f}]"
        )
