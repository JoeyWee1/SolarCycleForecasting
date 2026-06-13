#!/usr/bin/env python3
"""
run_gpr20.py
Batch runner: apply the GPR20 pipeline to every dataset in Data/mwd/ that
passes quality checks (>=100 points, span >=1 yr, max gap <10% of span).
One pickle file per star is written under --output_dir.

Usage
-----
python run_gpr20.py \
    --data_dir  /path/to/Data/mwd \
    --output_dir /path/to/rds/gpr20_results \
    [--star_type G] \
    [--skip_existing]
"""

import argparse
import logging
import pickle
import sys
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # no display on HPC

import numpy as np
import pandas as pd
import emcee
from scipy.optimize import minimize
from scipy.stats import norm

import celerite2
from celerite2 import terms

# ── repo root on path ─────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from helpers.df_ops import prepare_df, split_df, clean_df
from helpers.priors import find_classify_signals, get_priors

# ── quality thresholds (from GPR20 comments) ─────────────────────────────────
MIN_POINTS   = 100
MIN_SPAN_YRS = 1.0
MAX_GAP_FRAC = 0.10   # largest single gap must be < 10% of total span

# ── GPR20 pipeline functions (ported from notebook, no plots) ─────────────────

def set_params(log_params, k, gp):
    params = np.exp(log_params)
    gp.kernel = terms.SHOTerm(sigma=params[0], rho=params[k], Q=params[2*k])
    for i in range(1, k):
        gp.kernel += terms.SHOTerm(sigma=params[i], rho=params[k+i], Q=params[2*k+i])
    return gp

def NLL(log_params, gp, k, y):
    gp = set_params(log_params, k, gp)
    gp.recompute(quiet=True)
    return -gp.log_likelihood(y)

def lnPost(log_params, gp, k, y, initial_guesses, bounds, prior_std=3):
    if np.any(log_params < bounds[:, 0]) or np.any(log_params > bounds[:, 1]):
        return -np.inf
    gp = set_params(log_params, k, gp)
    gp.recompute(quiet=True)
    ll = gp.log_likelihood(y)
    if not np.isfinite(ll):
        return -np.inf
    return ll - 0.5 * np.sum(((np.exp(log_params) - np.exp(initial_guesses)) / prior_std) ** 2)

def check_constant(predictions, MAD, tol=2, percentile=84):
    amps = predictions.max(axis=1) - predictions.min(axis=1)
    return np.percentile(amps, percentile) / MAD < tol

def best_in_x(predictions, t, lookahead_years, t_start=None):
    t_start = t_start if t_start is not None else t[0]
    t_end   = min(t_start + lookahead_years * 365, t[-1])
    mask    = (t >= t_start) & (t < t_end)
    if not mask.any():
        mid = t[-1]
        return mid, (mid, mid)
    window_t    = t[mask]
    min_t       = window_t[np.argmin(predictions[:, mask], axis=1)]
    lb, hb      = np.percentile(min_t, [16, 84])
    return float(np.median(min_t)), (float(lb), float(hb))

def gaussian_smooth(t, y, sigma, n_eval=1000):
    t_eval = np.linspace(t.min(), t.max(), n_eval)
    w = np.exp(-0.5 * ((t[:, None] - t_eval[None, :]) / sigma) ** 2)
    return t_eval, (w * y[:, None]).sum(axis=0) / w.sum(axis=0)

def truth_in_x(data_df, year_ref, day_ref, t0_year, lookahead_years,
               pred_start_day=None, span_multiplier=0.04, recency_halflife_years=3):
    span  = data_df['day'].max() - data_df['day'].min()
    sigma = span * span_multiplier
    t_days = data_df['day'].values
    y      = data_df['sind'].values
    if pred_start_day is not None:
        halflife_days    = recency_halflife_years * 365
        recency_weights  = np.exp(-np.maximum(0, pred_start_day - t_days) * np.log(2) / halflife_days)
    else:
        recency_weights = np.ones(len(t_days))
    t_eval  = np.linspace(t_days.min(), t_days.max(), 1000)
    w       = np.exp(-0.5 * ((t_days[:, None] - t_eval[None, :]) / sigma) ** 2)
    w       = w * recency_weights[:, None]
    y_s     = (w * y[:, None]).sum(axis=0) / w.sum(axis=0)
    t_yr_s  = year_ref + (t_eval - day_ref) / 365
    mask    = (t_yr_s >= t0_year) & (t_yr_s <= t0_year + lookahead_years)
    if not mask.any():
        return np.nan
    return float(t_yr_s[mask][np.argmin(y_s[mask])])

def downsample_min_gap(df, minimum_gap):
    days = df["day"].values
    keep = np.zeros(len(days), dtype=bool)
    keep[0] = True
    last_kept = days[0]
    for i in range(1, len(days)):
        if days[i] - last_kept >= minimum_gap:
            keep[i] = True
            last_kept = days[i]
    return df[keep].reset_index(drop=True)

# ── quality filter ─────────────────────────────────────────────────────────────

def passes_checks(data_df, log):
    """Return True if the dataset meets the GPR20 requirements."""
    n = len(data_df)
    if n < MIN_POINTS:
        log.warning("SKIP: only %d datapoints (need %d)", n, MIN_POINTS)
        return False
    span = data_df['year'].iloc[-1] - data_df['year'].iloc[0]
    if span < MIN_SPAN_YRS:
        log.warning("SKIP: span %.2f yr (need %.1f)", span, MIN_SPAN_YRS)
        return False
    max_gap = data_df['year'].diff().dropna().max()
    if max_gap > span * MAX_GAP_FRAC:
        log.warning("SKIP: max gap %.2f yr > %.0f%% of span", max_gap, MAX_GAP_FRAC * 100)
        return False
    return True

# ── main pipeline ─────────────────────────────────────────────────────────────

def run_one_star(datapath, output_dir, star_type, log):
    """
    Run the full GPR20 pipeline for a single star and save results.
    Returns True on success, False if skipped/failed.
    """
    star_name = Path(datapath).stem.replace("_caii", "").upper()
    out_path  = Path(output_dir) / f"{star_name}_results.pkl"

    log.info("── %s ──", star_name)

    # ── params (mirrors GPR20 params cell) ────────────────────────────────────
    direct_bound_tol  = 0.1
    sm2016_bound_tol  = 0.2
    mean_bound_tol    = 1
    add_prefix        = False
    relative          = True
    error_percent     = 2.5
    sigma_upper_mult  = 5.0
    q_bounds_in       = [1, 5]
    require_mid       = True
    total_sample_count = 2500
    subsample         = 500
    n_walkers         = 32
    valid_metric      = "CRPS"
    SEED              = 1701
    n_windows_target  = 5
    pred_forward_years = 2
    low_cad           = 30
    high_cad          = 10
    min_gap           = 3
    minimum_gap       = 2.5

    # ── load data ─────────────────────────────────────────────────────────────
    raw_df  = pd.read_csv(datapath, sep=r'\s+', skip_blank_lines=True)
    data_df = prepare_df(raw_df, add_prefix=add_prefix, relative=relative)

    if not passes_checks(data_df, log):
        return False

    span = data_df['year'].iloc[-1] - data_df['year'].iloc[0]
    log.info("  %d points, span %.1f yr", len(data_df), span)

    # ── build window splits (mirrors GPR20 window-split cell) ─────────────────
    divide_train_splits = [(0.75 / n_windows_target) * n for n in range(1, n_windows_target + 1)]
    span_train_split    = (data_df['year'] < data_df['year'].iloc[0] + 0.15 * span).sum() / len(data_df)
    raw_train_splits    = np.maximum(divide_train_splits, span_train_split)
    train_splits        = np.unique(raw_train_splits)
    train_idxs          = np.round(len(data_df) * train_splits).astype(int)

    span_valid_splits = np.array([
        (data_df['year'] < data_df['year'].iloc[idx] + 0.2 * span).sum() / len(data_df) - ts
        for idx, ts in zip(train_idxs, train_splits)
    ])
    valid_splits  = np.maximum(0.1, span_valid_splits)
    valid_idxs    = np.round(len(data_df) * valid_splits).astype(int)

    retraining_split = train_splits + valid_splits
    mask             = retraining_split < 1
    train_splits     = train_splits[mask]
    valid_splits     = valid_splits[mask]
    retraining_split = retraining_split[mask]
    train_idxs       = train_idxs[mask]
    valid_idxs       = valid_idxs[mask]

    retraining_idxs  = train_idxs + valid_idxs
    mask_in_bounds   = retraining_idxs < len(data_df)
    retraining_idxs  = retraining_idxs[mask_in_bounds]
    train_idxs       = train_idxs[mask_in_bounds]
    valid_idxs       = valid_idxs[mask_in_bounds]
    train_splits     = train_splits[mask_in_bounds]
    valid_splits     = valid_splits[mask_in_bounds]
    test_splits      = (1 - retraining_split)[mask_in_bounds]

    if len(test_splits) < 1:
        test_split      = (data_df['year'] >= data_df['year'].iloc[-1] - 1).sum() / len(data_df)
        remaining       = 1 - test_split
        retrain_end_idx = int(np.round(len(data_df) * remaining)) - 1
        if data_df['year'].iloc[retrain_end_idx] - data_df['year'].iloc[0] < 1:
            log.warning("SKIP: combined train+valid span < 1 yr")
            return False
        log.warning("No valid splits found; using fallback 50/50 train/valid")
        train_splits    = np.array([remaining / 2])
        valid_splits    = np.array([remaining / 2])
        test_splits     = np.array([test_split])
        train_idxs      = np.round(len(data_df) * train_splits).astype(int)
        valid_idxs      = np.round(len(data_df) * valid_splits).astype(int)
        retraining_idxs = train_idxs + valid_idxs

    test_t0s       = data_df['year'].iloc[retraining_idxs - 1].to_numpy()
    t_end          = data_df['year'].iloc[-1]
    test_spans     = t_end - test_t0s
    test_window_lenss = [np.cumsum(np.ones(int(np.floor(ts)), dtype=int))
                         for ts in test_spans]

    log.info("  %d window split(s)", len(train_splits))

    # ── loop over splits ───────────────────────────────────────────────────────
    split_results = []

    for win_idx, (train_split, valid_split, test_split, test_window_lens) in enumerate(
            zip(train_splits, valid_splits, test_splits, test_window_lenss)):

        log.info("  window %d/%d  (train=%.2f valid=%.2f)",
                 win_idx + 1, len(train_splits), train_split, valid_split)

        data_df_ds = downsample_min_gap(data_df, min_gap)

        dirty_train_df, dirty_valid_df, test_df = split_df(
            data_df_ds, train_split=train_split, valid_split=valid_split)

        train_df, valid_df, MAD = clean_df(
            dirty_train_df, dirty_valid_df, tol=4, verbose=False, plot=False)

        # LSP priors
        classified_signal_data = find_classify_signals(
            dirty_train_df,
            plot_fitpeaks=False, verbose_fitpeaks=False,
            plot_genpriors=False, verbose_genpriors=False)

        rho_priors, rho_prior_bounds, _ = get_priors(
            classified_signal_data,
            star_type=star_type,
            direct_bound_tol=direct_bound_tol,
            sm2016_bound_tol=sm2016_bound_tol,
            mean_bound_tol=mean_bound_tol,
            verbose=False)

        # ── model comparison ──────────────────────────────────────────────────
        train_yerr = train_df['sind'] * error_percent / 100
        train_mean = train_df['sind'].mean()
        train_std  = train_df['sind'].std()

        prior_combos = {
            "1m":   {'k': 1, 'q_priors': None, 'cycle_keys': ['mid']},
            "2sm":  {'k': 2, 'q_priors': None, 'cycle_keys': ['short', 'mid']},
            "2ml":  {'k': 2, 'q_priors': None, 'cycle_keys': ['mid', 'long']},
            "3sml": {'k': 3, 'q_priors': None, 'cycle_keys': ['short', 'mid', 'long']},
        }
        if require_mid:
            prior_combos = {key: val for key, val in prior_combos.items()
                            if 'mid' in val['cycle_keys']}

        combo_results = {}

        for combo_name, combo_info in prior_combos.items():
            k            = combo_info['k']
            q_prior_type = combo_info['q_priors']
            cycle_keys   = combo_info['cycle_keys']

            np.random.seed(SEED)

            q_0s     = ([np.random.uniform(0, 0.5) for _ in range(k)]
                        if q_prior_type == 'overdamped'
                        else [np.random.uniform(0.5, 1) for _ in range(k)])
            sigma_0s = [train_std / k for _ in range(k)]
            rho_0s   = [rho_priors[ck] for ck in cycle_keys]

            initial_guess = np.concatenate([np.log(sigma_0s), np.log(rho_0s), np.log(q_0s)])

            q_bounds     = ([(-np.inf, np.log(0.5)) for _ in range(k)]
                            if q_prior_type == 'overdamped'
                            else [(np.log(q_bounds_in[0]), np.log(q_bounds_in[1])) for _ in range(k)])
            sigma_bounds = [(np.log(1e-4), np.log(train_std * sigma_upper_mult)) for _ in range(k)]
            rho_bounds   = np.log([rho_prior_bounds[ck] for ck in cycle_keys])
            bounds       = np.concatenate([sigma_bounds, rho_bounds, q_bounds])

            kernel = terms.SHOTerm(sigma=sigma_0s[0], rho=rho_0s[0], Q=q_0s[0])
            for i in range(1, k):
                kernel += terms.SHOTerm(sigma=sigma_0s[i], rho=rho_0s[i], Q=q_0s[i])

            gp = celerite2.GaussianProcess(kernel, mean=train_mean)
            gp.compute(train_df['day'], yerr=train_yerr)

            gp_results = minimize(NLL, initial_guess,
                                  args=(gp, k, train_df['sind'].to_numpy()),
                                  method='L-BFGS-B', bounds=bounds)

            gp = set_params(gp_results.x, k, gp)
            gp.recompute()
            best_params = np.exp(gp_results.x)

            mu, cov    = gp.predict(train_df['sind'], t=valid_df['day'], return_var=True)
            y_valid    = valid_df['sind'].to_numpy()
            valid_yerr = valid_df['sind'] * error_percent / 100
            total_std  = np.sqrt(cov + valid_yerr**2)
            total_var  = cov + valid_yerr**2

            nlpd = (0.5 * np.log(2 * np.pi * total_var)
                    + (y_valid - mu)**2 / (2 * total_var))
            mean_nlpd = nlpd.mean()

            z         = (y_valid - mu) / total_std
            crps      = total_std * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))
            mean_crps = crps.mean()

            combo_results[combo_name] = {
                "NLPD": mean_nlpd, "CRPS": mean_crps,
                "gp": gp, "params": best_params,
                "initial_guess": initial_guess, "bounds": bounds, "k": k,
            }

        # ── model selection ───────────────────────────────────────────────────
        best_combo_name = min(combo_results, key=lambda c: combo_results[c][valid_metric])
        best_result     = combo_results[best_combo_name]
        best_gp         = best_result['gp']
        best_params     = best_result['params']
        best_bounds     = best_result['bounds']
        k               = best_result['k']
        cycle_keys      = prior_combos[best_combo_name]['cycle_keys']

        log.info("    selected %s  (%s=%.4f)",
                 best_combo_name, valid_metric, best_result[valid_metric])

        # ── retraining ────────────────────────────────────────────────────────
        retrain_df   = downsample_min_gap(pd.concat([train_df, valid_df]), minimum_gap)
        retrain_yerr = retrain_df['sind'] * error_percent / 100
        retrain_mean = retrain_df['sind'].mean()

        initial_guess = np.log(best_params)
        params        = np.exp(initial_guess)
        kernel        = terms.SHOTerm(sigma=params[0], rho=params[k], Q=params[2*k])
        for i in range(1, k):
            kernel += terms.SHOTerm(sigma=params[i], rho=params[k+i], Q=params[2*k+i])

        gp_retrain = celerite2.GaussianProcess(kernel, mean=retrain_mean)
        gp_retrain.compute(retrain_df['day'], yerr=retrain_yerr)

        retrain_results = minimize(NLL, initial_guess,
                                   args=(gp_retrain, k, retrain_df['sind'].to_numpy()),
                                   method='L-BFGS-B', bounds=best_bounds)

        gp_retrain    = set_params(retrain_results.x, k, gp_retrain)
        gp_retrain.recompute()
        retrain_params = np.exp(retrain_results.x)

        # ── MCMC ─────────────────────────────────────────────────────────────
        np.random.seed(SEED)
        y = retrain_df['sind'].to_numpy()

        walker_start = np.log(retrain_params) + 1e-5 * np.random.randn(n_walkers, len(retrain_params))

        sampler = emcee.EnsembleSampler(
            n_walkers, len(retrain_params), lnPost,
            args=(gp_retrain, k, y, retrain_results.x, best_bounds))
        sampler.run_mcmc(walker_start, nsteps=total_sample_count, progress=False)

        ln_chains  = sampler.get_chain(discard=1000)
        ln_samples = ln_chains.reshape(-1, ln_chains.shape[-1])

        np.random.seed(SEED)
        idx        = np.random.choice(len(ln_samples), size=subsample, replace=False)
        ln_samples = ln_samples[idx]

        log.info("    MCMC done (%d samples after discard)", len(ln_samples))

        # ── predictions ───────────────────────────────────────────────────────
        t0_day  = retrain_df['day'].iloc[0]
        t0_year = retrain_df['year'].iloc[0]

        t_pred_start      = retrain_df['day'].iloc[-1]
        t_pred_start_year = retrain_df['year'].iloc[-1]
        t_ring_end        = t_pred_start + 365
        t_pred_end        = max(retrain_df['day'].iloc[-1] + 365 * pred_forward_years,
                                test_df['day'].iloc[-1])

        t_lookback_start = t_pred_start - 365 * 0.5
        sampled_days     = np.concatenate([
            np.arange(t_lookback_start, t_ring_end, low_cad),
            np.arange(t_ring_end, t_pred_end, high_cad),
        ])
        sampled_years = t0_year + (sampled_days - t0_day) / 365.25

        preds, pred_vars = [], []
        for ln_sample in ln_samples:
            set_params(ln_sample, k, gp_retrain)
            gp_retrain.recompute(quiet=True)
            pred, pred_var = gp_retrain.predict(retrain_df['sind'], t=sampled_days, return_var=True)
            preds.append(pred)
            pred_vars.append(pred_var)

        preds      = np.array(preds)
        pred_vars  = np.array(pred_vars)
        mean_pred  = preds.mean(axis=0)
        aleatoric  = pred_vars.mean(axis=0)
        epistemic  = preds.var(axis=0)
        total_std  = np.sqrt(aleatoric + epistemic)
        is_const   = check_constant(preds, MAD)

        def to_year(day):
            return t0_year + (day - t0_day) / 365

        test_window_lens = np.append(
            test_window_lens,
            data_df['year'].iloc[-1] - t_pred_start_year)

        best_ins = [
            best_in_x(preds, sampled_days, wl, t_start=t_pred_start)
            for wl in test_window_lens
        ]
        window_truths = [
            truth_in_x(data_df, t0_year, t0_day, t_pred_start_year, wl,
                       pred_start_day=t_pred_start)
            for wl in test_window_lens
        ]

        split_results.append({
            "star_name":        star_name,
            "win_idx":          win_idx,
            "train_split":      train_split,
            "valid_split":      valid_split,
            "best_combo_name":  best_combo_name,
            "best_params":      best_params,
            "mean_pred":        mean_pred,
            "sampled_days":     sampled_days,
            "sampled_years":    sampled_years,
            "aleatoric":        aleatoric,
            "epistemic":        epistemic,
            "total_std":        total_std,
            "is_constant":      is_const,
            "window_truths":    window_truths,
            "best_ins":         best_ins,
            "test_window_lens": test_window_lens,
            "t_pred_start_year": t_pred_start_year,
        })

    # ── save ──────────────────────────────────────────────────────────────────
    with open(out_path, "wb") as f:
        pickle.dump(split_results, f)
    log.info("  saved → %s", out_path)
    return True


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Batch GPR20 runner")
    parser.add_argument("--data_dir",    required=True,
                        help="Directory containing *_caii.txt files")
    parser.add_argument("--output_dir",  required=True,
                        help="Directory to write per-star pickle files")
    parser.add_argument("--star_type",   default="G",
                        help="Spectral type passed to get_priors (default: G)")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip stars whose output pickle already exists")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── logging: file + stdout ─────────────────────────────────────────────────
    log_path = output_dir / "run_gpr20.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout),
        ],
    )
    log = logging.getLogger("gpr20")
    log.info("data_dir   : %s", args.data_dir)
    log.info("output_dir : %s", output_dir)
    log.info("star_type  : %s", args.star_type)

    data_files = sorted(Path(args.data_dir).glob("*_caii.txt"))
    log.info("Found %d dataset files", len(data_files))

    n_ok = n_skip = n_fail = 0

    for i, datapath in enumerate(data_files):
        star_name = datapath.stem.replace("_caii", "").upper()
        out_path  = output_dir / f"{star_name}_results.pkl"

        if args.skip_existing and out_path.exists():
            log.info("[%d/%d] SKIP (exists): %s", i + 1, len(data_files), star_name)
            n_skip += 1
            continue

        log.info("[%d/%d] %s", i + 1, len(data_files), star_name)
        try:
            ok = run_one_star(str(datapath), output_dir, args.star_type, log)
            if ok:
                n_ok += 1
            else:
                n_skip += 1
        except Exception:
            log.error("FAILED: %s\n%s", star_name, traceback.format_exc())
            n_fail += 1

    log.info("Done.  processed=%d  skipped=%d  failed=%d", n_ok, n_skip, n_fail)


if __name__ == "__main__":
    main()
