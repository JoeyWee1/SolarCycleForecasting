"""
Tests for train_gpr on the three Mt. Wilson benchmark stars (mirrors GPR9).

Each star fixture runs the full training pipeline once (7-model comparison with
L-BFGS-B) and caches the result for all tests that use it.

Performance criterion
---------------------
The best model's mean NLPD on the held-out validation set must be below
NLPD_THRESHOLD = -0.5.  GPR9 records HD201091 at -1.23; the threshold gives
~0.73 units of slack to account for numerical variance across platforms.

Run slow tests with:  pytest -m slow
Skip slow tests with: pytest -m "not slow"
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

import celerite2
from celerite2 import terms as celerite_terms

from helpers.gpr import train_gpr
from helpers.df_ops import prepare_df, split_df, clean_df

DATA_DIR = Path(__file__).parents[1] / "Data" / "benchmark"

BENCHMARK_STARS = {
    "HD81809": {
        "path":      DATA_DIR / "HD81809_Mt_wilson_data.txt",
        "star_type": "G",
    },
    "HD160346": {
        "path":      DATA_DIR / "HD160346_Mt_wilson_data.txt",
        "star_type": "K",
    },
    "HD201091": {
        "path":      DATA_DIR / "HD201091_Mt_wilson_data.txt",
        "star_type": "K",
    },
}

# GPR9 recorded HD201091 at -1.23; threshold gives ~0.73 units of slack
NLPD_THRESHOLD = -0.5


def _compute_validation_nlpd(gp, datapath, train_split=0.8, valid_split=0.19,
                              error_percent=2.5):
    """
    Replicate train_gpr's data split and compute the mean NLPD of *gp* on the
    validation set.  Mirrors the evaluation logic inside train_gpr exactly so
    the returned value is directly comparable to what train_gpr selects on.
    """
    raw_df = pd.read_csv(str(datapath), sep=r"\s+", skip_blank_lines=True)
    data_df = prepare_df(raw_df, relative=True)
    dirty_train, dirty_valid, _ = split_df(data_df, train_split=train_split,
                                           valid_split=valid_split)
    train_df, valid_df, _ = clean_df(dirty_train, dirty_valid, tol=4,
                                     verbose=False, plot=False)

    mu, cov = gp.predict(train_df["sind"], t=valid_df["day"].to_numpy(),
                         return_var=True)
    y_valid    = valid_df["sind"].to_numpy()
    valid_yerr = y_valid * error_percent / 100
    total_var  = cov + valid_yerr ** 2
    nlpd = (0.5 * np.log(2 * np.pi * total_var)
            + (y_valid - mu) ** 2 / (2 * total_var))
    return nlpd.mean(), train_df, valid_df


@pytest.fixture(scope="module", params=list(BENCHMARK_STARS.keys()))
def trained_gpr(request):
    """
    Run train_gpr on one benchmark star.  Scoped to module so training happens
    once per star even when multiple tests reference this fixture.
    Returns (star_name, best_gp, train_df, valid_df, nlpd).
    """
    star_name = request.param
    info = BENCHMARK_STARS[star_name]

    best_gp = train_gpr(
        datapath        = str(info["path"]),
        star_type       = info["star_type"],
        star_name       = star_name,
        verbose         = False,
        plot            = False,
        loop_verbose    = False,
        loop_plot       = False,
        loop_savefigs   = False,
        results_verbose = False,
        results_plot    = False,
    )

    nlpd, train_df, valid_df = _compute_validation_nlpd(best_gp, info["path"])
    return star_name, best_gp, train_df, valid_df, nlpd


# ─────────────────────────────────────────────────────────────────────────────
# Structural / type tests
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.slow
def test_returns_gaussian_process(trained_gpr):
    """train_gpr must return a celerite2 GaussianProcess, not None."""
    _, best_gp, _, _, _ = trained_gpr
    assert isinstance(best_gp, celerite2.GaussianProcess), (
        f"Expected celerite2.GaussianProcess, got {type(best_gp)}"
    )


def _collect_sho_terms(kernel):
    """Recursively collect all leaf SHOTerm components from a (possibly nested) kernel."""
    if isinstance(kernel, celerite_terms.SHOTerm):
        return [kernel]
    if hasattr(kernel, "terms"):
        leaves = []
        for t in kernel.terms:
            leaves.extend(_collect_sho_terms(t))
        return leaves
    return []


@pytest.mark.slow
def test_has_valid_kernel(trained_gpr):
    """The returned GP must have a non-None kernel composed of SHOTerm components."""
    _, best_gp, _, _, _ = trained_gpr
    assert best_gp.kernel is not None
    leaf_terms = _collect_sho_terms(best_gp.kernel)
    assert len(leaf_terms) >= 1, "Expected at least one SHOTerm in the kernel"
    for t in leaf_terms:
        assert isinstance(t, celerite_terms.SHOTerm), (
            f"Expected SHOTerm leaf, got {type(t)}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Prediction quality tests
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.slow
def test_predictions_are_finite(trained_gpr):
    """GP predictions and variances on the validation set must all be finite."""
    _, best_gp, train_df, valid_df, _ = trained_gpr
    mu, var = best_gp.predict(
        train_df["sind"],
        t=valid_df["day"].to_numpy(),
        return_var=True,
    )
    assert np.all(np.isfinite(mu)),  "GP mean predictions contain non-finite values"
    assert np.all(np.isfinite(var)), "GP prediction variances contain non-finite values"


@pytest.mark.slow
def test_prediction_variances_are_positive(trained_gpr):
    """All prediction variances must be strictly positive."""
    _, best_gp, train_df, valid_df, _ = trained_gpr
    _, var = best_gp.predict(
        train_df["sind"],
        t=valid_df["day"].to_numpy(),
        return_var=True,
    )
    assert np.all(var > 0), (
        f"Non-positive prediction variances detected (min = {var.min():.2e})"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Performance test (core regression test)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.slow
def test_validation_nlpd_below_threshold(trained_gpr):
    """
    The best model's mean NLPD on the validation set must be below NLPD_THRESHOLD.

    GPR9 benchmark: HD201091 achieves -1.23.  Threshold = -0.5 gives 0.73 units
    of slack for numerical differences across platforms.
    """
    star_name, _, _, _, nlpd = trained_gpr
    assert nlpd < NLPD_THRESHOLD, (
        f"{star_name}: validation NLPD = {nlpd:.4f}, "
        f"threshold = {NLPD_THRESHOLD} "
        f"(GPR9 benchmark: HD201091 achieved -1.23)"
    )


@pytest.mark.slow
def test_training_log_likelihood_is_finite(trained_gpr):
    """The GP's log-likelihood on the training data must be a finite number."""
    star_name, best_gp, train_df, _, _ = trained_gpr
    ll = best_gp.log_likelihood(train_df["sind"].to_numpy())
    assert np.isfinite(ll), (
        f"{star_name}: training log-likelihood is {ll} (expected finite value)"
    )
