#!/usr/bin/env python3
"""
cadence_analysis.py
Batch runner: apply the cadence analysis lookahead pipeline to every simulated star in
Data/simulated/<rate>/ across all (or a specified) sampling-rate folders.

Results are written as per-star pickle files so the run is fully resumable.
After each folder is complete a summary CSV is saved; a combined CSV is written
at the end.

Usage
-----
# All folders, all 100 stars, 5 windows:
python run_gpr24b.py --sim_root Data/simulated --out_dir Results/simulated

# One folder only (useful for parallelising over folders on HPC):
python run_gpr24b.py --sim_root Data/simulated --out_dir Results/simulated \
    --rate 1.0d --n_windows 10

# Quick test: 3 G-type stars, 5 windows:
python run_gpr24b.py --sim_root Data/simulated --out_dir Results/simulated \
    --n_stars 3 --star_type_filter G
"""

import argparse
import logging
import pickle
import sys
import traceback
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import emcee
from scipy.optimize import minimize
from scipy.stats import norm
import celerite2
from celerite2 import terms

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from helpers.df_ops import prepare_df, split_df, clean_df
from helpers.priors import find_classify_signals, get_priors

# ── GP helpers (from GPR24b) ──────────────────────────────────────────────────

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

def lnPost_gp(log_params, gp, k, y, initial_guesses, bounds, prior_std=3):
    if np.any(log_params < bounds[:, 0]) or np.any(log_params > bounds[:, 1]):
        return -np.inf
    gp = set_params(log_params, k, gp)
    gp.recompute(quiet=True)
    lnL = gp.log_likelihood(y)
    if not np.isfinite(lnL):
        return -np.inf
    return lnL - 0.5 * np.sum(((np.exp(log_params) - np.exp(initial_guesses)) / prior_std) ** 2)

def downsample_min_gap(df, minimum_gap):
    days = df['day'].values
    keep = np.zeros(len(days), dtype=bool)
    keep[0] = True
    last = days[0]
    for i in range(1, len(days)):
        if days[i] - last >= minimum_gap:
            keep[i] = True
            last = days[i]
    return df[keep].reset_index(drop=True)

def check_constant(predictions, MAD, tol=2, percentile=84):
    amps = predictions.max(axis=1) - predictions.min(axis=1)
    return np.percentile(amps, percentile) / MAD < tol

# ── Fourier ground truth (from GPR24b) ───────────────────────────────────────

def fit_Fourier(raw_data_df, MAD, med_ref, rho_priors, test_df,
                error_percent, n_walkers, subsample=500):
    fourier_df = raw_data_df[(raw_data_df['sind'] - med_ref).abs() < MAD].reset_index(drop=True)
    raw_std = np.std(fourier_df['sind'])

    log_AB_bounds = np.log([[1e-4, 5 * raw_std] for _ in range(6)])
    log_T_bounds  = np.log([[1e-4, raw_data_df['day'].iloc[-1] - raw_data_df['day'].iloc[0]]])
    c_bounds      = [[fourier_df['sind'].min(), fourier_df['sind'].max()]]
    bounds        = np.concatenate([log_AB_bounds, log_T_bounds, c_bounds])

    t0_ref_day  = raw_data_df['day'].iloc[0]
    t0_ref_year = raw_data_df['year'].iloc[0]
    def to_year(day): return t0_ref_year + (day - t0_ref_day) / 365.25

    def lnPost_Fourier(params, t, y, sigma, weights, bounds):
        if np.any(params < bounds[:, 0]) or np.any(params > bounds[:, 1]):
            return -np.inf
        A1, A2, A3 = np.exp(params[0:3])
        B1, B2, B3 = np.exp(params[3:6])
        T = np.exp(params[6]); c = params[7]
        model  = c + A1*np.sin(2*np.pi*t/T) + B1*np.cos(2*np.pi*t/T)
        model += A2*np.sin(4*np.pi*t/T)      + B2*np.cos(4*np.pi*t/T)
        model += A3*np.sin(6*np.pi*t/T)      + B3*np.cos(6*np.pi*t/T)
        return np.sum(weights * (-np.log(sigma) - 0.5*np.log(2*np.pi)
                                 - 0.5*((y - model)/sigma)**2))

    ig_AB = [(fourier_df['sind'].max() - fourier_df['sind'].min()) / 6] * 6
    ig    = np.concatenate([np.log(ig_AB), np.log([rho_priors['mid']]),
                            [np.median(fourier_df['sind'])]])
    wsc   = ig + 1e-4 * np.random.randn(n_walkers, len(ig))

    tau      = rho_priors['mid']
    t0_day   = test_df['day'].iloc[0]
    t_arr    = fourier_df['day'].to_numpy()
    y_arr    = fourier_df['sind'].to_numpy()
    sigma    = y_arr * error_percent / 100
    weights  = np.where(t_arr < t0_day, np.exp(-(t0_day - t_arr) / tau), 1.0)

    sampler = emcee.EnsembleSampler(n_walkers, len(ig), lnPost_Fourier,
                                    args=(t_arr, y_arr, sigma, weights, bounds))
    sampler.run_mcmc(wsc, nsteps=10000, progress=False)
    flat = sampler.get_chain(discard=5000, flat=True)

    t_plot      = np.linspace(raw_data_df['day'].min(), raw_data_df['day'].max(), 2000)
    t_plot_year = to_year(t_plot)
    step        = max(1, len(flat) // subsample)
    model_preds = []
    for s in flat[::step]:
        A1, A2, A3 = np.exp(s[0:3]); B1, B2, B3 = np.exp(s[3:6])
        T = np.exp(s[6]); c = s[7]
        m  = c + A1*np.sin(2*np.pi*t_plot/T) + B1*np.cos(2*np.pi*t_plot/T)
        m += A2*np.sin(4*np.pi*t_plot/T)      + B2*np.cos(4*np.pi*t_plot/T)
        m += A3*np.sin(6*np.pi*t_plot/T)      + B3*np.cos(6*np.pi*t_plot/T)
        model_preds.append(m)
    model_preds = np.array(model_preds)
    return (model_preds, model_preds.mean(0),
            np.percentile(model_preds, 16, 0),
            np.percentile(model_preds, 84, 0),
            t_plot_year)

def truth_in_x(model_preds, t_plot_year, t0_year, lookahead_years):
    mask         = (t_plot_year >= t0_year) & (t_plot_year <= t0_year + lookahead_years)
    window_preds = model_preds[:, mask]
    window_t     = t_plot_year[mask]
    min_years    = window_t[np.argmin(window_preds, axis=1)]
    lb, ub       = np.percentile(min_years, [16, 84])
    return float(np.median(min_years)), (float(lb), float(ub))

# ── run_star (from GPR24b) ────────────────────────────────────────────────────

def run_star(datapath, star_name, star_type='G', add_prefix=False,
             error_percent=2.5, sigma_upper_mult=5.0, q_bounds_in=(1, 5),
             n_target_windows=5, min_gap=3, n_walkers=32,
             total_sample_count=2500, subsample=500, SEED=1701,
             pred_forward_years=2, low_cad=30, high_cad=10,
             direct_bound_tol=0.1, sm2016_bound_tol=0.2, mean_bound_tol=1,
             require_mid=True, valid_metric='CRPS', verbose=False):
    np.random.seed(SEED)

    try:
        raw_df  = pd.read_csv(datapath, sep=r'\s+', skip_blank_lines=True)
        data_df = prepare_df(raw_df, add_prefix=add_prefix, relative=True)
    except Exception as e:
        return {'star': star_name, 'skipped': True, 'skip_reason': f'load error: {e}', 'splits': []}

    n_obs = len(data_df)
    if n_obs < 100:
        return {'star': star_name, 'n_obs': n_obs, 'skipped': True,
                'skip_reason': 'fewer than 100 observations', 'splits': []}

    span = data_df['year'].iloc[-1] - data_df['year'].iloc[0]
    if span < 1:
        return {'star': star_name, 'n_obs': n_obs, 'span_years': span,
                'skipped': True, 'skip_reason': 'span < 1 year', 'splits': []}

    data_df = downsample_min_gap(data_df, min_gap)

    n_win              = n_target_windows
    divide_train_splits = [(0.75 / n_win) * n for n in range(1, n_win + 1)]
    span_train_split   = (data_df['year'] < data_df['year'].iloc[0] + 0.15 * span).sum() / len(data_df)
    raw_train_splits   = np.maximum(divide_train_splits, span_train_split)
    train_splits       = np.unique(raw_train_splits)
    train_idxs         = np.round(len(data_df) * train_splits).astype(int)

    span_valid_splits = np.array([
        (data_df['year'] < data_df['year'].iloc[idx] + 0.2 * span).sum() / len(data_df) - ts
        for idx, ts in zip(train_idxs, train_splits)
    ])
    valid_splits      = np.maximum(0.1, span_valid_splits)
    retraining_split  = train_splits + valid_splits
    mask              = retraining_split < 1
    train_splits      = train_splits[mask]
    valid_splits      = valid_splits[mask]
    retraining_split  = retraining_split[mask]

    retraining_idxs = np.round(len(data_df) * retraining_split).astype(int)
    mask_ib         = retraining_idxs < len(data_df)
    train_splits    = train_splits[mask_ib]
    valid_splits    = valid_splits[mask_ib]
    retraining_idxs = retraining_idxs[mask_ib]

    if len(train_splits) < 1:
        remaining       = 1 - (data_df['year'] >= data_df['year'].iloc[-1] - 1).sum() / len(data_df)
        retrain_end_idx = int(np.round(len(data_df) * remaining)) - 1
        if data_df['year'].iloc[retrain_end_idx] - data_df['year'].iloc[0] < 1:
            return {'star': star_name, 'n_obs': len(data_df), 'span_years': span,
                    'skipped': True, 'skip_reason': 'combined train+valid span < 1 year', 'splits': []}
        train_splits    = np.array([remaining / 2])
        valid_splits    = np.array([remaining / 2])
        retraining_idxs = np.round(len(data_df) * (train_splits + valid_splits)).astype(int)

    test_t0s         = data_df['year'].iloc[retraining_idxs - 1].to_numpy()
    test_spans       = data_df['year'].iloc[-1] - test_t0s
    test_window_lenss = [np.cumsum(np.ones(max(0, int(np.floor(ts))), dtype=int))
                         for ts in test_spans]

    prior_combos = {
        '1m':   {'k': 1, 'q_priors': None, 'cycle_keys': ['mid']},
        '2sm':  {'k': 2, 'q_priors': None, 'cycle_keys': ['short', 'mid']},
        '2ml':  {'k': 2, 'q_priors': None, 'cycle_keys': ['mid', 'long']},
        '3sml': {'k': 3, 'q_priors': None, 'cycle_keys': ['short', 'mid', 'long']},
    }
    if require_mid:
        prior_combos = {nm: v for nm, v in prior_combos.items() if 'mid' in v['cycle_keys']}

    split_results = []

    for train_split, valid_split, test_window_lens in zip(
            train_splits, valid_splits, test_window_lenss):
        try:
            dirty_train_df, dirty_valid_df, test_df = split_df(
                data_df, train_split=train_split, valid_split=valid_split)
            train_df, valid_df, MAD = clean_df(
                dirty_train_df, dirty_valid_df, tol=4, verbose=False, plot=False)

            classified_signal_data = find_classify_signals(
                dirty_train_df,
                manual_freq ='log',
                plot_fitpeaks=False, verbose_fitpeaks=False,
                plot_genpriors=False, verbose_genpriors=False)
            rho_priors, rho_prior_bounds, _ = get_priors(
                classified_signal_data, star_type=star_type,
                direct_bound_tol=direct_bound_tol,
                sm2016_bound_tol=sm2016_bound_tol,
                mean_bound_tol=mean_bound_tol,
                verbose=False)

            train_yerr = train_df['sind'] * error_percent / 100
            train_mean = train_df['sind'].mean()
            train_std  = train_df['sind'].std()

            combo_results = {}
            for combo_name, combo_info in prior_combos.items():
                k_c          = combo_info['k']
                q_prior_type = combo_info['q_priors']
                cycle_keys_c = combo_info['cycle_keys']

                np.random.seed(SEED)
                q_0s     = ([np.random.uniform(0, 0.5) for _ in range(k_c)]
                            if q_prior_type == 'overdamped'
                            else [np.random.uniform(0.5, 1) for _ in range(k_c)])
                sigma_0s = [train_std / k_c for _ in range(k_c)]
                rho_0s   = [rho_priors[ck] for ck in cycle_keys_c]

                ig_raw   = np.concatenate([sigma_0s, rho_0s, q_0s])
                q_bnd    = ([(- np.inf, np.log(0.5)) for _ in range(k_c)]
                            if q_prior_type == 'overdamped'
                            else [(np.log(q_bounds_in[0]), np.log(q_bounds_in[1]))
                                  for _ in range(k_c)])
                sigma_bnd = [(np.log(1e-4), np.log(train_std * sigma_upper_mult))
                             for _ in range(k_c)]
                rho_bnd   = np.log([rho_prior_bounds[ck] for ck in cycle_keys_c])
                bounds_c  = np.concatenate([sigma_bnd, rho_bnd, q_bnd])

                perturbs  = ig_raw * 0.1 * np.random.normal(size=(25, len(ig_raw)))
                perturbed = np.log(np.clip(ig_raw + perturbs, 1e-6, None))

                best_NLL, best_res_c = np.inf, None
                for ig in perturbed:
                    kernel = terms.SHOTerm(sigma=sigma_0s[0], rho=rho_0s[0], Q=q_0s[0])
                    for ki in range(1, k_c):
                        kernel += terms.SHOTerm(sigma=sigma_0s[ki], rho=rho_0s[ki], Q=q_0s[ki])
                    gp = celerite2.GaussianProcess(kernel, mean=train_mean)
                    gp.compute(train_df['day'], yerr=train_yerr)
                    res = minimize(NLL, ig, args=(gp, k_c, train_df['sind'].to_numpy()),
                                   method='L-BFGS-B', bounds=bounds_c)
                    if res.fun < best_NLL:
                        best_NLL, best_res_c = res.fun, res

                gp = set_params(best_res_c.x, k_c, gp)
                gp.recompute()
                bp = np.exp(best_res_c.x)

                mu, cov    = gp.predict(train_df['sind'], t=valid_df['day'], return_var=True)
                y_valid    = valid_df['sind'].to_numpy()
                valid_yerr = valid_df['sind'] * error_percent / 100
                total_var  = cov + valid_yerr ** 2
                total_std  = np.sqrt(total_var)

                nlpd = (0.5 * np.log(2 * np.pi * total_var)
                        + (y_valid - mu) ** 2 / (2 * total_var)).mean()
                z    = (y_valid - mu) / total_std
                crps = (total_std * (z * (2 * norm.cdf(z) - 1)
                        + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))).mean()

                combo_results[combo_name] = {'NLPD': nlpd, 'CRPS': crps,
                                             'params': bp, 'bounds': bounds_c}

            best_metric, best_combo_name = np.inf, None
            for cname, cr in combo_results.items():
                if cr[valid_metric] < best_metric:
                    best_metric, best_combo_name = cr[valid_metric], cname
                    best_params = cr['params']
                    best_bounds = cr['bounds']

            if verbose:
                logging.getLogger('gpr24b').info(
                    '  %s: selected %s (%s=%.4f)', star_name, best_combo_name,
                    valid_metric, best_metric)

            retrain_df   = downsample_min_gap(pd.concat([train_df, valid_df]), min_gap)
            combo_info   = prior_combos[best_combo_name]
            k            = combo_info['k']
            cycle_keys   = combo_info['cycle_keys']
            retrain_yerr = retrain_df['sind'] * error_percent / 100
            retrain_mean = retrain_df['sind'].mean()

            p0     = best_params
            kernel = terms.SHOTerm(sigma=p0[0], rho=p0[k], Q=p0[2*k])
            for ki in range(1, k):
                kernel += terms.SHOTerm(sigma=p0[ki], rho=p0[k + ki], Q=p0[2*k + ki])
            gp_retrain = celerite2.GaussianProcess(kernel, mean=retrain_mean)
            gp_retrain.compute(retrain_df['day'], yerr=retrain_yerr)

            retrain_res = minimize(NLL, np.log(best_params),
                                   args=(gp_retrain, k, retrain_df['sind'].to_numpy()),
                                   method='L-BFGS-B', bounds=best_bounds)
            gp_retrain   = set_params(retrain_res.x, k, gp_retrain)
            gp_retrain.recompute()
            retrain_params = np.exp(retrain_res.x)

            np.random.seed(SEED)
            y   = retrain_df['sind'].to_numpy()
            wsc = np.log(retrain_params) + 1e-5 * np.random.randn(n_walkers, len(retrain_params))
            sampler = emcee.EnsembleSampler(
                n_walkers, len(retrain_params), lnPost_gp,
                args=(gp_retrain, k, y, retrain_res.x, best_bounds))
            sampler.run_mcmc(wsc, nsteps=total_sample_count, progress=False)

            ln_chains  = sampler.get_chain(discard=1000)
            ln_samples = ln_chains.reshape(-1, ln_chains.shape[-1])
            np.random.seed(SEED)
            sel        = np.random.choice(len(ln_samples), size=subsample, replace=False)
            ln_samples = ln_samples[sel]

            t0_day            = retrain_df['day'].iloc[0]
            t0_year           = retrain_df['year'].iloc[0]
            t_pred_start      = retrain_df['day'].iloc[-1]
            t_pred_start_year = retrain_df['year'].iloc[-1]
            t_ring_end        = t_pred_start + 365
            t_pred_end        = max(t_pred_start + 365 * pred_forward_years,
                                    test_df['day'].iloc[-1])

            sampled_days  = np.concatenate([
                np.arange(t_pred_start - 365 * 0.5, t_ring_end, low_cad),
                np.arange(t_ring_end, t_pred_end, high_cad),
            ])
            sampled_years = t0_year + (sampled_days - t0_day) / 365.25

            preds, pred_vars = [], []
            for ln_s in ln_samples:
                set_params(ln_s, k, gp_retrain)
                gp_retrain.recompute(quiet=True)
                p, pv = gp_retrain.predict(retrain_df['sind'], t=sampled_days, return_var=True)
                preds.append(p); pred_vars.append(pv)

            preds     = np.array(preds)
            pred_vars = np.array(pred_vars)
            aleatoric = pred_vars.mean(axis=0)
            epistemic = preds.var(axis=0)
            is_const  = check_constant(preds, MAD)

            fourier_preds, _, _, _, fourier_t_year = fit_Fourier(
                raw_data_df=data_df, MAD=MAD,
                med_ref=dirty_train_df['sind'].median(),
                rho_priors=rho_priors, test_df=test_df,
                error_percent=error_percent, n_walkers=n_walkers)

            test_window_lens = np.append(
                test_window_lens, data_df['year'].iloc[-1] - t_pred_start_year)

            t0_idx        = np.searchsorted(sampled_years, t_pred_start_year)
            ale_t0        = float(aleatoric[t0_idx]) if t0_idx < len(aleatoric) else np.nan
            epi_t0        = float(epistemic[t0_idx]) if t0_idx < len(epistemic) else np.nan
            tot_var_t0    = ale_t0 + epi_t0
            aleatoric_frac = ale_t0 / tot_var_t0 if tot_var_t0 > 0 else np.nan

            mid_idx       = cycle_keys.index('mid')
            rho_mid_years = retrain_params[k + mid_idx] / 365.25
            Q_mid         = retrain_params[2*k + mid_idx]
            n_cycles_obs  = span / rho_mid_years

            windows = []
            for w_len in test_window_lens:
                t_end_w = t_pred_start_year + float(w_len)
                w_mask  = (sampled_years >= t_pred_start_year) & (sampled_years <= t_end_w)
                if not w_mask.any():
                    windows.append({
                        'lookahead_years': float(w_len), 't0_year': float(t_pred_start_year),
                        **{kk: np.nan for kk in [
                            'gpr_med', 'gpr_lb68', 'gpr_ub68', 'gpr_lb95', 'gpr_ub95',
                            'truth_med', 'truth_lb', 'truth_ub',
                            'error', 'abs_error', 'gpr_width_68', 'truth_width']},
                        'in_68': False, 'in_95': False})
                    continue

                w_preds       = preds[:, w_mask]
                w_years       = sampled_years[w_mask]
                gpr_min_years = w_years[np.argmin(w_preds, axis=1)]
                gpr_med       = float(np.median(gpr_min_years))
                p2_5, p16, p84, p97_5 = np.percentile(gpr_min_years, [2.5, 16, 84, 97.5])

                try:
                    truth_med, (truth_lb, truth_ub) = truth_in_x(
                        fourier_preds, fourier_t_year, t_pred_start_year, float(w_len))
                except Exception:
                    truth_med = truth_lb = truth_ub = np.nan

                error = gpr_med - truth_med if not np.isnan(truth_med) else np.nan
                windows.append({
                    'lookahead_years': float(w_len), 't0_year': float(t_pred_start_year),
                    'gpr_med': gpr_med,
                    'gpr_lb68': float(p16), 'gpr_ub68': float(p84),
                    'gpr_lb95': float(p2_5), 'gpr_ub95': float(p97_5),
                    'truth_med': float(truth_med) if not np.isnan(truth_med) else np.nan,
                    'truth_lb':  float(truth_lb)  if not np.isnan(truth_lb)  else np.nan,
                    'truth_ub':  float(truth_ub)  if not np.isnan(truth_ub)  else np.nan,
                    'error':     float(error)      if not np.isnan(error)     else np.nan,
                    'abs_error': float(abs(error)) if not np.isnan(error)     else np.nan,
                    'in_68': bool(p16 <= truth_med <= p84)   if not np.isnan(truth_med) else False,
                    'in_95': bool(p2_5 <= truth_med <= p97_5) if not np.isnan(truth_med) else False,
                    'gpr_width_68': float(p84 - p16),
                    'truth_width':  float(truth_ub - truth_lb) if not np.isnan(truth_ub) else np.nan,
                })

            split_results.append({
                'best_combo': best_combo_name,
                'valid_metric_value': float(best_metric),
                'rho_mid_years': float(rho_mid_years),
                'Q_mid': float(Q_mid),
                'n_cycles_obs': float(n_cycles_obs),
                'is_const': bool(is_const),
                'aleatoric_frac_at_t0': float(aleatoric_frac) if not np.isnan(aleatoric_frac) else np.nan,
                'windows': windows,
            })

        except Exception as e:
            logging.getLogger('gpr24b').warning('%s: split error (%s)', star_name, e)
            continue

    return {'star': star_name, 'n_obs': len(data_df), 'span_years': float(span),
            'skipped': False, 'skip_reason': None, 'splits': split_results}


def results_to_df(results_list):
    rows = []
    for r in results_list:
        if r.get('skipped', False):
            continue
        for split_idx, split in enumerate(r['splits']):
            base = {
                'star': r['star'], 'n_obs': r['n_obs'], 'span_years': r['span_years'],
                'split_idx': split_idx,
                'best_combo': split['best_combo'],
                'rho_mid_years': split['rho_mid_years'],
                'Q_mid': split['Q_mid'],
                'n_cycles_obs': split['n_cycles_obs'],
                'is_const': split['is_const'],
                'aleatoric_frac_at_t0': split['aleatoric_frac_at_t0'],
                'valid_metric_value': split['valid_metric_value'],
            }
            for win in split['windows']:
                rows.append({**base, **win})
    return pd.DataFrame(rows)


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Batch GPR24b runner on simulated data')
    parser.add_argument('--sim_root',         default='Data/simulated',
                        help='Root of simulated data (contains <rate>d/ subfolders)')
    parser.add_argument('--out_dir',          default='Results/simulated',
                        help='Output directory; per-star pickles and per-rate CSVs written here')
    parser.add_argument('--n_windows',        type=int, default=5,
                        help='Target number of train/valid windows per star (default: 5)')
    parser.add_argument('--rate',             default=None,
                        help='Run only this folder, e.g. "1.0d" (default: all folders)')
    parser.add_argument('--n_stars',          type=int, default=None,
                        help='Limit number of stars per folder (default: all)')
    parser.add_argument('--star_type_filter', default=None,
                        help='Only run stars of this type, e.g. "G" (default: all)')
    parser.add_argument('--skip_existing',    action='store_true',
                        help='Skip stars whose output pickle already exists')
    parser.add_argument('--verbose',          action='store_true')
    args = parser.parse_args()

    sim_root = Path(args.sim_root)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path = out_dir / 'run_gpr24b.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s  %(levelname)-7s  %(message)s',
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],
    )
    log = logging.getLogger('gpr24b')
    log.info('sim_root  : %s', sim_root)
    log.info('out_dir   : %s', out_dir)
    log.info('n_windows : %d', args.n_windows)

    params      = np.load(sim_root / 'star_params.npz', allow_pickle=True)
    star_labels = params['labels']          # (N,) string array, e.g. 'G', 'ME'

    def prior_type(stype): return stype  # pass full type; get_priors has ME and MM keys

    # Select folders
    all_folders = sorted(
        [f for f in sim_root.iterdir() if f.is_dir()],
        key=lambda f: float(f.name.rstrip('d'))
    )
    folders = [f for f in all_folders if args.rate is None or f.name == args.rate]
    if not folders:
        log.error('No folders matched --rate %s', args.rate)
        sys.exit(1)

    # Select star indices
    indices = np.arange(len(star_labels))
    if args.star_type_filter:
        indices = indices[star_labels[indices] == args.star_type_filter]
    if args.n_stars is not None:
        indices = indices[:args.n_stars]

    log.info('Folders: %s', [f.name for f in folders])
    log.info('Stars  : %d', len(indices))

    all_dfs = []

    for folder in folders:
        rate_str  = folder.name
        rate_days = float(rate_str.rstrip('d'))
        pkl_dir   = out_dir / rate_str
        pkl_dir.mkdir(exist_ok=True)
        log.info('=== %s ===', rate_str)

        n_ok = n_skip = n_fail = 0

        for i in indices:
            stype      = star_labels[i]
            star_name  = f'sim{i:04d}_{stype}'
            pkl_path   = pkl_dir / f'{star_name}.pkl'
            fpath      = folder / f'{star_name}_caii.txt'

            if args.skip_existing and pkl_path.exists():
                log.info('  SKIP (exists): %s', star_name)
                n_skip += 1
                continue

            if not fpath.exists():
                log.warning('  MISSING: %s', fpath)
                n_fail += 1
                continue

            log.info('  %s  (P_cyc=%.1f yr, P_rot=%.1f d)',
                     star_name, params['P_cycs'][i], params['P_rots'][i])
            try:
                result = run_star(
                    datapath=str(fpath),
                    star_name=star_name,
                    star_type=prior_type(stype),
                    add_prefix=False,
                    n_target_windows=args.n_windows,
                    verbose=args.verbose,
                )
                with open(pkl_path, 'wb') as f:
                    pickle.dump(result, f)
                n_ok += 1
            except Exception:
                log.error('  FAILED: %s\n%s', star_name, traceback.format_exc())
                n_fail += 1

        log.info('%s done — ok=%d  skipped=%d  failed=%d', rate_str, n_ok, n_skip, n_fail)

        # Combine all pickles for this folder into a CSV
        results = []
        for pkl_path in sorted(pkl_dir.glob('*.pkl')):
            with open(pkl_path, 'rb') as f:
                results.append(pickle.load(f))

        if results:
            df = results_to_df(results)
            df['sampling_rate_days'] = rate_days
            csv_path = out_dir / f'{rate_str}.csv'
            df.to_csv(csv_path, index=False)
            log.info('  CSV → %s  (%d rows)', csv_path, len(df))
            all_dfs.append(df)

    # Combined CSV across all rates
    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined_path = out_dir / 'all_rates.csv'
        combined.to_csv(combined_path, index=False)
        log.info('Combined CSV → %s  (%d rows)', combined_path, len(combined))

    log.info('All done.')


if __name__ == '__main__':
    main()
