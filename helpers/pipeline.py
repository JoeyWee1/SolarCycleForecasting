import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import emcee
from scipy.optimize import minimize
from scipy.stats import norm
import celerite2
from celerite2 import terms

from helpers.df_ops import prepare_df, split_df, clean_df, downsample_min_gap
from helpers.priors import find_classify_signals, get_priors
from helpers.gpr import set_params, NLL
from helpers.MCMC import lnPost_gp
from helpers.eval import check_constant, best_in_x, fit_Fourier, truth_in_x


def star_window_analysis(datapath, star_name, star_type='G', add_prefix=False,
             error_percent=2.5, sigma_upper_mult=5.0, q_bounds_in=(1, 5),
             n_target_windows=5, min_gap=3, n_walkers=32,
             total_sample_count=2500, subsample=500, SEED=1701,
             pred_forward_years=2, low_cad=30, high_cad=10,
             direct_bound_tol=0.1, sm2016_bound_tol=0.2, mean_bound_tol=1,
             require_mid=True, valid_metric='CRPS', verbose=False):
    '''
    Runs the full GPR + Fourier ground truth pipeline on one star across multiple train/valid splits.
    Flatten the output with results_to_df().

    In:
        datapath (str): path to the whitespace-separated data file
        star_name (str): identifier used in the output dict
        star_type (str): spectral type, one of F, G, K, ME, MM
        add_prefix (bool): add 2400000 to JD if the file uses truncated Julian dates
        error_percent (float): assumed percentage measurement error on S-index
        sigma_upper_mult (float): upper bound on sigma as a multiple of the training std
        q_bounds_in (tuple): (lower, upper) Q bounds for underdamped SHO terms
        n_target_windows (int): target number of train/valid splits
        min_gap (float): minimum gap in days between observations after downsampling
        n_walkers (int): number of emcee walkers
        total_sample_count (int): total MCMC steps per run
        subsample (int): number of MCMC draws to retain for prediction
        SEED (int): random seed
        pred_forward_years (float): minimum years to predict beyond the training data
        low_cad / high_cad (float): day spacing for the near/far prediction grid
        direct_bound_tol / sm2016_bound_tol / mean_bound_tol (float): prior bound tolerances
        require_mid (bool): if True, discard kernel combos without a mid-range term
        valid_metric (str): 'CRPS' or 'NLPD', used for model selection
        verbose (bool): whether to print progress

    Out:
        result (dict): nested structure with keys
            star, n_obs, span_years, skipped, skip_reason
            splits (list of dicts), each with keys
                best_combo, valid_metric_value, rho_mid_years, Q_mid,
                n_cycles_obs, is_const, aleatoric_frac_at_t0
                windows (list of dicts), each with keys
                    lookahead_years, t0_year,
                    gpr_med, gpr_lb68, gpr_ub68, gpr_lb95, gpr_ub95,
                    truth_med, truth_lb, truth_ub,
                    error, abs_error, in_68, in_95, gpr_width_68, truth_width
    '''
    # Reproducible results!!!
    np.random.seed(SEED)

    #---Data Loading---
    try:
        raw_df = pd.read_csv(datapath, sep=r'\s+', skip_blank_lines=True)
        data_df = prepare_df(raw_df, add_prefix=add_prefix, relative=True)
    except Exception as e: # Fails gracefully
        return {'star': star_name, 'skipped': True, 'skip_reason': f'load error: {e}', 'splits': []}

    # Skip stars with too few observations
    n_obs = len(data_df)
    if n_obs < 100:
        return {'star': star_name, 'n_obs': n_obs, 'skipped': True, # Log failures
                'skip_reason': 'fewer than 100 observations', 'splits': []}

    # Skip stars with too short a time span
    span = data_df['year'].iloc[-1] - data_df['year'].iloc[0]
    if span < 1:
        return {'star': star_name, 'n_obs': n_obs, 'span_years': span,
                'skipped': True, 'skip_reason': 'span < 1 year', 'splits': []}

    data_df = downsample_min_gap(data_df, min_gap)

    #---Build Windows---
    # Evenly spaced train split fractions across data
    n_win = n_target_windows
    divide_train_splits = [(0.75 / n_win) * n for n in range(1, n_win + 1)]
    span_train_split = ((data_df['year'] < data_df['year'].iloc[0] + 0.15 * span).sum() / len(data_df))
    raw_train_splits = np.maximum(divide_train_splits, span_train_split)
    train_splits = np.unique(raw_train_splits)
    train_idxs = np.round(len(data_df) * train_splits).astype(int)

    # Validation split sized to span approx 20% of total time
    span_valid_splits = np.array([
        (data_df['year'] < data_df['year'].iloc[idx] + 0.2 * span).sum() / len(data_df) - ts
        for idx, ts in zip(train_idxs, train_splits)
    ])
    valid_splits = np.maximum(0.1, span_valid_splits)

    # Drop splits where train+valid would exhaust the data
    retraining_split = train_splits + valid_splits
    mask = retraining_split < 1
    train_splits = train_splits[mask]
    valid_splits = valid_splits[mask]
    train_idxs = train_idxs[mask]
    retraining_split = retraining_split[mask]

    # Compute retraining end indices; fallback to single split if none remain
    retraining_idxs = np.round(len(data_df) * retraining_split).astype(int)
    mask_ib = retraining_idxs < len(data_df)
    train_splits = train_splits[mask_ib]
    valid_splits = valid_splits[mask_ib]
    retraining_idxs = retraining_idxs[mask_ib]

    if len(train_splits) < 1:
        test_split = (data_df['year'] >= data_df['year'].iloc[-1] - 1).sum() / len(data_df)
        remaining = 1 - test_split
        retrain_end_idx = int(np.round(len(data_df) * remaining)) - 1
        if data_df['year'].iloc[retrain_end_idx] - data_df['year'].iloc[0] < 1:
            return {'star': star_name, 'n_obs': len(data_df), 'span_years': span,
                    'skipped': True, 'skip_reason': 'combined train+valid span < 1 year', 'splits': []}
        train_splits = np.array([remaining / 2])
        valid_splits = np.array([remaining / 2])
        retraining_idxs = np.round(len(data_df) * (train_splits + valid_splits)).astype(int)

    # Per-split test start times and integer lookahead lengths
    test_t0s = data_df['year'].iloc[retraining_idxs - 1].to_numpy()
    test_spans = data_df['year'].iloc[-1] - test_t0s
    test_window_lenss = [
        np.cumsum(np.ones(max(0, int(np.floor(ts))), dtype=int))
        for ts in test_spans
    ]

    #---Model Selection---
    prior_combos = {
        '1m':   {'k': 1, 'q_priors': None, 'cycle_keys': ['mid']},
        '2sm':  {'k': 2, 'q_priors': None, 'cycle_keys': ['short', 'mid']},
        '2ml':  {'k': 2, 'q_priors': None, 'cycle_keys': ['mid',   'long']},
        '3sml': {'k': 3, 'q_priors': None, 'cycle_keys': ['short', 'mid', 'long']},
    }
    if require_mid:
        prior_combos = {nm: v for nm, v in prior_combos.items() if 'mid' in v['cycle_keys']}

    # Collect per-split results
    split_results = []

    for train_split, valid_split, test_window_lens in zip(
            train_splits, valid_splits, test_window_lenss):

        try:
            # Rebuild window splits
            dirty_train_df, dirty_valid_df, test_df = split_df(data_df, train_split=train_split, valid_split=valid_split)
            train_df, valid_df, MAD = clean_df(dirty_train_df, dirty_valid_df, tol=4, verbose=False, plot=False)

            #Find priors
            classified_signal_data = find_classify_signals(
                dirty_train_df,
                plot_fitpeaks=False, verbose_fitpeaks=False,
                plot_genpriors=False, verbose_genpriors=False)
            rho_priors, rho_prior_bounds, _ = get_priors(
                classified_signal_data, star_type=star_type,
                direct_bound_tol=direct_bound_tol,
                sm2016_bound_tol=sm2016_bound_tol,
                mean_bound_tol=mean_bound_tol,
                verbose=False)

            #---Compare each combination---
            # Training set summary statistics
            train_yerr = train_df['sind'] * error_percent / 100
            train_mean = train_df['sind'].mean()
            train_std = train_df['sind'].std()

            combo_results = {}
            for combo_name, combo_info in prior_combos.items():
                k_c = combo_info['k']
                q_prior_type = combo_info['q_priors']
                cycle_keys_c = combo_info['cycle_keys']

                # Initial parameter guesses
                np.random.seed(SEED)
                q_0s = ([np.random.uniform(0, 0.5) for _ in range(k_c)]
                        if q_prior_type == 'overdamped' # Not used but there for future generality
                        else [np.random.uniform(0.5, 1) for _ in range(k_c)])
                sigma_0s = [train_std / k_c for _ in range(k_c)]
                rho_0s = [rho_priors[ck] for ck in cycle_keys_c]

                # Parameter bounds
                ig_raw = np.concatenate([sigma_0s, rho_0s, q_0s])
                q_bnd = ([(- np.inf, np.log(0.5)) for _ in range(k_c)]
                         if q_prior_type == 'overdamped'
                         else [(np.log(q_bounds_in[0]), np.log(q_bounds_in[1]))
                               for _ in range(k_c)])
                sigma_bnd = [(np.log(1e-4), np.log(train_std * sigma_upper_mult))
                             for _ in range(k_c)]
                rho_bnd = np.log([rho_prior_bounds[ck] for ck in cycle_keys_c])
                bounds_c = np.concatenate([sigma_bnd, rho_bnd, q_bnd])

                # Perturbed restarts for multi-start optimisation
                perturbs = ig_raw * 0.1 * np.random.normal(size=(25, len(ig_raw)))
                perturbed = np.log(np.clip(ig_raw + perturbs, 1e-6, None))

                # Best L-BFGS-B result across all restarts
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

                # Extract best-fit parameters
                gp = set_params(best_res_c.x, k_c, gp)
                gp.recompute()
                bp = np.exp(best_res_c.x)

                # Predict on validation set
                mu, cov = gp.predict(train_df['sind'], t=valid_df['day'], return_var=True)
                y_valid = valid_df['sind'].to_numpy()
                valid_yerr = valid_df['sind'] * error_percent / 100
                total_var = cov + valid_yerr ** 2
                total_std = np.sqrt(total_var)

                # Compute NLPD and CRPS
                nlpd = (0.5 * np.log(2 * np.pi * total_var)
                        + (y_valid - mu) ** 2 / (2 * total_var)).mean()
                z = (y_valid - mu) / total_std
                crps = (total_std * (z * (2 * norm.cdf(z) - 1)
                        + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))).mean()

                # Store combo metrics
                combo_results[combo_name] = {
                    'NLPD': nlpd, 'CRPS': crps,
                    'params': bp, 'bounds': bounds_c,
                }

            #---Select best model---
            best_metric, best_combo_name = np.inf, None
            for cname, cr in combo_results.items():
                if cr[valid_metric] < best_metric:
                    best_metric, best_combo_name = cr[valid_metric], cname
                    best_params = cr['params']
                    best_bounds = cr['bounds']

            if verbose:
                print(f"{star_name}: selected {best_combo_name} ({valid_metric}={best_metric:.4f})")

            #---Retrain on train+valid---
            retrain_df = downsample_min_gap(pd.concat([train_df, valid_df]), min_gap)
            combo_info = prior_combos[best_combo_name]
            k = combo_info['k']
            cycle_keys = combo_info['cycle_keys']

            retrain_yerr = retrain_df['sind'] * error_percent / 100
            retrain_mean = retrain_df['sind'].mean()

            p0 = best_params
            kernel = terms.SHOTerm(sigma=p0[0], rho=p0[k], Q=p0[2*k])
            for ki in range(1, k):
                kernel += terms.SHOTerm(sigma=p0[ki], rho=p0[k + ki], Q=p0[2*k + ki])

            gp_retrain = celerite2.GaussianProcess(kernel, mean=retrain_mean)
            gp_retrain.compute(retrain_df['day'], yerr=retrain_yerr)

            retrain_res = minimize(NLL, np.log(best_params),
                                   args=(gp_retrain, k, retrain_df['sind'].to_numpy()),
                                   method='L-BFGS-B', bounds=best_bounds)
            gp_retrain = set_params(retrain_res.x, k, gp_retrain)
            gp_retrain.recompute()
            retrain_params = np.exp(retrain_res.x)

            #---MCMC posterior---
            np.random.seed(SEED)
            y = retrain_df['sind'].to_numpy()
            wsc = np.log(retrain_params) + 1e-5 * np.random.randn(n_walkers, len(retrain_params))

            sampler = emcee.EnsembleSampler(
                n_walkers, len(retrain_params), lnPost_gp,
                args=(gp_retrain, k, y, retrain_res.x, best_bounds))
            sampler.run_mcmc(wsc, nsteps=total_sample_count, progress=False)

            ln_chains = sampler.get_chain(discard=1000)
            ln_samples = ln_chains.reshape(-1, ln_chains.shape[-1])
            np.random.seed(SEED)
            sel = np.random.choice(len(ln_samples), size=subsample, replace=False)
            ln_samples = ln_samples[sel]

            #---Forecast grid---
            t0_day = retrain_df['day'].iloc[0]
            t0_year = retrain_df['year'].iloc[0]

            t_pred_start = retrain_df['day'].iloc[-1]
            t_pred_start_year = retrain_df['year'].iloc[-1]
            t_ring_end = t_pred_start + 365
            t_pred_end = max(t_pred_start + 365 * pred_forward_years, test_df['day'].iloc[-1])

            sampled_days = np.concatenate([
                np.arange(t_pred_start - 365 * 0.5, t_ring_end, low_cad),
                np.arange(t_ring_end, t_pred_end, high_cad)])
            sampled_years = t0_year + (sampled_days - t0_day) / 365.25

            # Generate MCMC predictive ensemble
            preds, pred_vars = [], []
            for ln_s in ln_samples:
                set_params(ln_s, k, gp_retrain)
                gp_retrain.recompute(quiet=True)
                p, pv = gp_retrain.predict(retrain_df['sind'], t=sampled_days, return_var=True)
                preds.append(p); pred_vars.append(pv)

            # Decompose variance into aleatoric and epistemic
            preds = np.array(preds)
            pred_vars = np.array(pred_vars)
            aleatoric = pred_vars.mean(axis=0)
            epistemic = preds.var(axis=0)
            is_const = check_constant(preds, MAD)

            #---Fourier ground truth---
            fourier_preds, _, _, _, fourier_t_year = fit_Fourier(
                raw_data_df=data_df, MAD=MAD,
                med_ref=dirty_train_df['sind'].median(),
                rho_priors=rho_priors, test_df=test_df,
                error_percent=error_percent, n_walkers=n_walkers,
            )

            # Append full-span window
            test_window_lens = np.append(
                test_window_lens, data_df['year'].iloc[-1] - t_pred_start_year)

            #---Split-level diagnostics---
            t0_idx = np.searchsorted(sampled_years, t_pred_start_year)
            ale_t0 = float(aleatoric[t0_idx]) if t0_idx < len(aleatoric) else np.nan
            epi_t0 = float(epistemic[t0_idx]) if t0_idx < len(epistemic) else np.nan
            tot_var_t0 = ale_t0 + epi_t0
            aleatoric_frac = ale_t0 / tot_var_t0 if tot_var_t0 > 0 else np.nan

            # Extract mid-cycle period and quality factor
            mid_idx = cycle_keys.index('mid')
            rho_mid_days = retrain_params[k + mid_idx]
            rho_mid_years = rho_mid_days / 365.25
            Q_mid = retrain_params[2*k + mid_idx]
            n_cycles_obs = span / rho_mid_years

            #---Per-window statistics---
            windows = []
            for w_len in test_window_lens:
                t_end_w = t_pred_start_year + float(w_len)
                w_mask = (sampled_years >= t_pred_start_year) & (sampled_years <= t_end_w)

                if not w_mask.any():
                    windows.append({
                        'lookahead_years': float(w_len),
                        't0_year': float(t_pred_start_year),
                        **{kk: np.nan for kk in [
                            'gpr_med', 'gpr_lb68', 'gpr_ub68', 'gpr_lb95', 'gpr_ub95',
                            'truth_med', 'truth_lb', 'truth_ub',
                            'error', 'abs_error', 'gpr_width_68', 'truth_width',
                        ]},
                        'in_68': False, 'in_95': False,
                    })
                    continue

                w_preds = preds[:, w_mask]
                w_years = sampled_years[w_mask]
                gpr_min_years = w_years[np.argmin(w_preds, axis=1)]
                gpr_med = float(np.median(gpr_min_years))
                p2_5, p16, p84, p97_5 = np.percentile(gpr_min_years, [2.5, 16, 84, 97.5])

                try:
                    truth_med, (truth_lb, truth_ub) = truth_in_x(
                        fourier_preds, fourier_t_year, t_pred_start_year, float(w_len))
                except Exception:
                    truth_med = np.nan
                    truth_lb = np.nan
                    truth_ub = np.nan

                error = gpr_med - truth_med if not np.isnan(truth_med) else np.nan
                windows.append({
                    'lookahead_years': float(w_len),
                    't0_year': float(t_pred_start_year),
                    'gpr_med': gpr_med,
                    'gpr_lb68': float(p16),
                    'gpr_ub68': float(p84),
                    'gpr_lb95': float(p2_5),
                    'gpr_ub95': float(p97_5),
                    'truth_med': float(truth_med) if not np.isnan(truth_med) else np.nan,
                    'truth_lb': float(truth_lb) if not np.isnan(truth_lb) else np.nan,
                    'truth_ub': float(truth_ub) if not np.isnan(truth_ub) else np.nan,
                    'error': float(error) if not np.isnan(error) else np.nan,
                    'abs_error': float(abs(error)) if not np.isnan(error) else np.nan,
                    'in_68': bool(p16 <= truth_med <= p84) if not np.isnan(truth_med) else False,
                    'in_95': bool(p2_5 <= truth_med <= p97_5) if not np.isnan(truth_med) else False,
                    'gpr_width_68': float(p84 - p16),
                    'truth_width': float(truth_ub - truth_lb) if not np.isnan(truth_ub) else np.nan,
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
            if verbose:
                print(f"{star_name}: split error ({e})")
            continue

    return {
        'star': star_name,
        'n_obs': len(data_df),
        'span_years': float(span),
        'skipped': False,
        'skip_reason': None,
        'splits': split_results,
    }

def results_to_df(results_list):
    '''
    Flattens a list of star_window_analysis() dicts into a per-(star, split, window) DataFrame.
    Skipped stars are excluded.

    In:
        results_list (list of dicts): output of star_window_analysis(), one entry per star

    Out:
        df (DataFrame): one row per (star, split, lookahead window) with columns
            star, n_obs, span_years, split_idx, best_combo, rho_mid_years, Q_mid,
            n_cycles_obs, is_const, aleatoric_frac_at_t0, valid_metric_value,
            lookahead_years, t0_year, gpr_med, gpr_lb68, gpr_ub68, gpr_lb95, gpr_ub95,
            truth_med, truth_lb, truth_ub, error, abs_error, in_68, in_95,
            gpr_width_68, truth_width
    '''
    rows = []
    for r in results_list:
        if r.get('skipped', False): # Do not include skipped stars
            continue
        for split_idx, split in enumerate(r['splits']):
            base = {
                'star': r['star'],
                'n_obs': r['n_obs'],
                'span_years': r['span_years'],
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
                rows.append({**base, **win}) # Append the merged dictionary
    return pd.DataFrame(rows)

def run_star(datapath, star_name, star_type='G', add_prefix=False,
             error_percent=2.5, sigma_upper_mult=5.0, q_bounds_in=(1, 5),
             min_gap=3, n_walkers=32, total_sample_count=2500, subsample=500, SEED=1701,
             pred_forward_years=2, high_cad=10,
             direct_bound_tol=0.1, sm2016_bound_tol=0.2, mean_bound_tol=1,
             require_mid=True, valid_metric='CRPS', lookahead_years=3,
             verbose=False, plot=False):
    '''
    Runs the full GPR on one star using a fixed 75/25 train/validation split with no test set.
    *THIS CREATES FORWARD PREDICTIONS FOR THE LOOKAHEAD YEARS*
    
    In:
        datapath (str): path to the data file
        star_name (str): identifier used in plot titles
        star_type (str): spectral type, one of F, G, K, ME, MM
        add_prefix (bool): add 2400000 to JD if the file uses truncated Julian dates
        error_percent (float): assumed percentage measurement error on S-index
        sigma_upper_mult (float): upper bound on sigma as a multiple of the training std
        q_bounds_in (tuple): (lower, upper) Q bounds for underdamped SHO terms
        min_gap (float): minimum gap in days between observations after downsampling
        n_walkers (int): number of emcee walkers
        total_sample_count (int): total MCMC steps
        subsample (int): number of MCMC draws to retain for prediction
        SEED (int): random seed
        pred_forward_years (float): minimum years to predict beyond the training data
        high_cad (float): day spacing for the prediction grid
        direct_bound_tol / sm2016_bound_tol / mean_bound_tol (float): prior bound tolerances
        require_mid (bool): if True, discard kernel combos without a mid-range term
        valid_metric (str): 'CRPS' or 'NLPD', used for model selection
        lookahead_years (float or array): lookahead window(s) to evaluate in years
        verbose (bool): whether to print progress
        plot (bool): whether to plot the GPR predictions and best-in-x markers

    Out:
        result (dict): keys
            preds (ndarray): shape (subsample, n_times) MCMC prediction draws
            sampled_years (ndarray): shape (n_times,) year grid for predictions
            best_in_x (dict): keyed by lookahead value; each entry is {med, lb, ub} in years
    '''
    # Reproducibility
    np.random.seed(SEED)

    # Normalise lookahead_years to array in case a float is entered
    lookaheads = np.atleast_1d(lookahead_years)

    #---Data Loading---
    raw_df = pd.read_csv(datapath, sep=r'\s+', skip_blank_lines=True)
    data_df = prepare_df(raw_df, add_prefix=add_prefix, relative=True)
    # Remove closely-spaced observations
    data_df = downsample_min_gap(data_df, min_gap)

    #---Split 75/25---
    dirty_train_df, dirty_valid_df, _ = split_df(data_df, train_split=0.75, valid_split=0.25)
    train_df, valid_df, MAD = clean_df(dirty_train_df, dirty_valid_df, tol=4, verbose=verbose, plot=False)

    #---Priors---
    classified_signal_data = find_classify_signals(
        dirty_train_df,
        plot_fitpeaks=False, verbose_fitpeaks=False,
        plot_genpriors=False, verbose_genpriors=False)
    rho_priors, rho_prior_bounds, _ = get_priors(
        classified_signal_data, star_type=star_type,
        direct_bound_tol=direct_bound_tol, sm2016_bound_tol=sm2016_bound_tol,
        mean_bound_tol=mean_bound_tol, verbose=verbose)

    #---Model Selection---
    prior_combos = {
        '1m':   {'k': 1, 'q_priors': None, 'cycle_keys': ['mid']},
        '2sm':  {'k': 2, 'q_priors': None, 'cycle_keys': ['short', 'mid']},
        '2ml':  {'k': 2, 'q_priors': None, 'cycle_keys': ['mid', 'long']},
        '3sml': {'k': 3, 'q_priors': None, 'cycle_keys': ['short', 'mid', 'long']},
    }
    if require_mid:
        prior_combos = {nm: v for nm, v in prior_combos.items() if 'mid' in v['cycle_keys']}

    # Training set summary statistics
    train_yerr = train_df['sind'] * error_percent / 100
    train_mean = train_df['sind'].mean()
    train_std = train_df['sind'].std()

    combo_results = {}
    for combo_name, combo_info in prior_combos.items():
        k = combo_info['k']
        q_prior_type = combo_info['q_priors']
        cycle_keys = combo_info['cycle_keys']

        # Initial parameter guesses
        np.random.seed(SEED)
        q_0s = ([np.random.uniform(0, 0.5) for _ in range(k)]
                if q_prior_type == 'overdamped'
                else [np.random.uniform(0.5, 1) for _ in range(k)])
        sigma_0s = [train_std / k for _ in range(k)]
        rho_0s = [rho_priors[ck] for ck in cycle_keys]

        # Parameter bounds
        ig_raw = np.concatenate([sigma_0s, rho_0s, q_0s])
        q_bounds = ([(- np.inf, np.log(0.5)) for _ in range(k)]
                    if q_prior_type == 'overdamped'
                    else [(np.log(q_bounds_in[0]), np.log(q_bounds_in[1])) for _ in range(k)])
        sigma_bounds = [(np.log(1e-4), np.log(train_std * sigma_upper_mult)) for _ in range(k)]
        rho_bounds = np.log([rho_prior_bounds[ck] for ck in cycle_keys])
        bounds = np.concatenate([sigma_bounds, rho_bounds, q_bounds])

        # Perturbed restarts for multi-start optimisation
        perturbs = ig_raw * 0.1 * np.random.normal(size=(25, len(ig_raw)))
        perturbed = np.log(np.clip(ig_raw + perturbs, 1e-6, None))

        # Best L-BFGS-B result across all restarts
        best_NLL, best_res = np.inf, None
        for ig in perturbed:
            kernel = terms.SHOTerm(sigma=sigma_0s[0], rho=rho_0s[0], Q=q_0s[0])
            for ki in range(1, k):
                kernel += terms.SHOTerm(sigma=sigma_0s[ki], rho=rho_0s[ki], Q=q_0s[ki])
            gp = celerite2.GaussianProcess(kernel, mean=train_mean)
            gp.compute(train_df['day'], yerr=train_yerr)
            res = minimize(NLL, ig, args=(gp, k, train_df['sind'].to_numpy()),
                           method='L-BFGS-B', bounds=bounds)
            if res.fun < best_NLL:
                best_NLL, best_res = res.fun, res

        # Extract best-fit parameters
        gp = set_params(best_res.x, k, gp)
        gp.recompute()
        bp = np.exp(best_res.x)

        # Predict on validation set
        mu, cov = gp.predict(train_df['sind'], t=valid_df['day'], return_var=True)
        y_valid = valid_df['sind'].to_numpy()
        valid_yerr = valid_df['sind'] * error_percent / 100
        total_var = cov + valid_yerr ** 2
        total_std = np.sqrt(total_var)

        # Compute NLPD and CRPS
        nlpd = (0.5 * np.log(2 * np.pi * total_var) + (y_valid - mu) ** 2 / (2 * total_var)).mean()
        z = (y_valid - mu) / total_std
        crps = (total_std * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))).mean()

        # Store combo metrics
        combo_results[combo_name] = {'NLPD': nlpd, 'CRPS': crps, 'params': bp, 'bounds': bounds}

    # Select best model by chosen metric
    best_metric, best_combo_name = np.inf, None
    for cname, cr in combo_results.items():
        if cr[valid_metric] < best_metric:
            best_metric, best_combo_name = cr[valid_metric], cname
            best_params = cr['params']
            best_bounds = cr['bounds']

    if verbose:
        print(f"{star_name}: selected {best_combo_name} ({valid_metric}={best_metric:.4f})")

    #---Retrain on all data---
    retrain_df = downsample_min_gap(pd.concat([train_df, valid_df]), min_gap)
    k = prior_combos[best_combo_name]['k']
    retrain_yerr = retrain_df['sind'] * error_percent / 100
    retrain_mean = retrain_df['sind'].mean()

    # Build kernel from best params and refit
    p0 = best_params
    kernel = terms.SHOTerm(sigma=p0[0], rho=p0[k], Q=p0[2*k])
    for ki in range(1, k):
        kernel += terms.SHOTerm(sigma=p0[ki], rho=p0[k + ki], Q=p0[2*k + ki])

    gp_retrain = celerite2.GaussianProcess(kernel, mean=retrain_mean)
    gp_retrain.compute(retrain_df['day'], yerr=retrain_yerr)

    retrain_res = minimize(NLL, np.log(best_params),
                           args=(gp_retrain, k, retrain_df['sind'].to_numpy()),
                           method='L-BFGS-B', bounds=best_bounds)
    gp_retrain = set_params(retrain_res.x, k, gp_retrain)
    gp_retrain.recompute()
    retrain_params = np.exp(retrain_res.x)

    #---MCMC Posterior---
    # Initialise walkers near MAP solution
    np.random.seed(SEED)
    y = retrain_df['sind'].to_numpy()
    wsc = np.log(retrain_params) + 1e-5 * np.random.randn(n_walkers, len(retrain_params))
    sampler = emcee.EnsembleSampler(
        n_walkers, len(retrain_params), lnPost_gp,
        args=(gp_retrain, k, y, retrain_res.x, best_bounds))
    sampler.run_mcmc(wsc, nsteps=total_sample_count, progress=False)

    # Discard burn-in and subsample chains
    ln_chains = sampler.get_chain(discard=1000)
    ln_samples = ln_chains.reshape(-1, ln_chains.shape[-1])
    np.random.seed(SEED)
    sel = np.random.choice(len(ln_samples), size=subsample, replace=False)
    ln_samples = ln_samples[sel]

    #---Forecast Grid---
    t0_day = retrain_df['day'].iloc[0]
    t0_year = retrain_df['year'].iloc[0]
    t_pred_start = retrain_df['day'].iloc[-1]
    n_days_forward = 365 * max(pred_forward_years, float(np.max(lookaheads)))
    sampled_days = np.arange(t_pred_start, t_pred_start + n_days_forward, high_cad)
    sampled_years = t0_year + (sampled_days - t0_day) / 365.25

    # Generate MCMC predictive ensemble
    preds = []
    for ln_s in ln_samples:
        set_params(ln_s, k, gp_retrain)
        gp_retrain.recompute(quiet=True)
        p, _ = gp_retrain.predict(retrain_df['sind'], t=sampled_days, return_var=True)
        preds.append(p)
    preds = np.array(preds)

    #---Best in X---
    to_year = lambda d: t0_year + (d - t0_day) / 365.25
    bix = {}
    for la in lookaheads:
        med, (lb, ub) = best_in_x(preds, sampled_days, la, t_start=t_pred_start)
        bix[float(la)] = {'med': to_year(med), 'lb': to_year(lb), 'ub': to_year(ub)}

    #---Plot---
    if plot:
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.scatter(retrain_df['year'], retrain_df['sind'], s=5, color='steelblue', label='Data', zorder=3)
        mean_pred = preds.mean(axis=0)
        lb68 = np.percentile(preds, 16, axis=0)
        ub68 = np.percentile(preds, 84, axis=0)
        ax.plot(sampled_years, mean_pred, color='C1', label='GPR mean')
        ax.fill_between(sampled_years, lb68, ub68, color='C1', alpha=0.3, label='68%')
        colors = plt.cm.autumn(np.linspace(0, 0.8, len(lookaheads)))
        for la, color in zip(lookaheads, colors):
            b = bix[float(la)]
            ax.axvline(b['med'], color=color, lw=1.2, label=f'{la}yr: {b["med"]:.2f}')
            ax.axvspan(b['lb'], b['ub'], color=color, alpha=0.1)
        ax.set_xlabel('Year')
        ax.set_ylabel('S-index')
        ax.set_title(star_name)
        ax.legend()
        plt.tight_layout()
        plt.show()

    return {
        'preds': preds,
        'sampled_years': sampled_years,
        'best_in_x': bix,
    }