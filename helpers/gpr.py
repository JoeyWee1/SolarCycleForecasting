import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import minimize

import celerite2
from celerite2 import terms

from prettytable import PrettyTable

from helpers.df_ops import prepare_df, split_df, clean_df
from helpers.priors import find_classify_signals, get_priors

def set_params(log_params, k, gp):
    '''
    Sets the kernel of a celerite2 GP from a flat log-parameter vector.
    Parameter order is [log_sigma x k, log_rho x k, log_Q x k].

    In:
        log_params (array of floats): log-space GP parameters, length 3*k
        k (int): number of SHO terms
        gp (celerite2 GaussianProcess): the GP whose kernel will be updated

    Out:
        gp (celerite2 GaussianProcess): the same GP with its kernel updated in place
    '''
    params = np.exp(log_params)
    gp.kernel = terms.SHOTerm(sigma=params[0], rho=params[k], Q=params[2*k])
    for i in range(1, k):
        gp.kernel += terms.SHOTerm(sigma=params[i], rho=params[k+i], Q=params[2*k+i])
    return gp

def NLL(log_params, gp, k, y):
    '''
    Computes the negative log-likelihood of the GP at the given log-space parameters.

    In:
        log_params (array of floats): log-space GP parameters, length 3*k
        gp (celerite2 GaussianProcess): the GP to evaluate
        k (int): number of SHO terms
        y (array of floats): S-index observations

    Out:
        NLL (float): negative log-likelihood
    '''
    gp = set_params(log_params, k, gp)
    gp.recompute(quiet=True)
    return -gp.log_likelihood(y)


def train_gpr(
        datapath,
        direct_bound_tol = 0.1, sm2016_bound_tol = 0.2,
        mean_bound_tol = 1,
        add_prefix: bool = False,
        train_split: float = 0.8, valid_split: float = 0.19,
        star_type: str = None, star_name: str = None,
        error_percent: float = 2.5,
        sigma_upper_mult: float = 5.0, q_bounds_in = [5, 100],
        relative: bool = True,
        verbose = True, plot = True,
        loop_verbose: bool = False, loop_plot: bool = False, loop_savefigs: bool = True,
        results_verbose: bool = True, results_plot: bool = True, ax = None,
        result_plot_cadence = 1,
        result_plot_extra = 100,
        require_mid: bool = True,
        SEED = 1701
        ):
        '''
        Single-star GPR pipeline. Loads data, runs model selection across all kernel
        combos (1s, 1m, 1l, 2sm, 2ml, 2sl, 3sml, const), selects the best by NLPD on the
        validation set, and optionally plots the results.

        This is not the final pipeline. It was the baseline GPR pipeline and is now deprecated.
        It is frequently used in older notebools and is therefore included here.
        For the current pipeline, see the pipeline submodule.

        In:
            datapath (str): path to the whitespace-separated data file
            star_type (str): spectral type, one of F, G, K, ME, MM
            star_name (str): used for plot titles
            train_split (float): fraction of rows for training
            valid_split (float): fraction of rows for validation
            error_percent (float): assumed percentage measurement error on S-index
            sigma_upper_mult (float): upper bound on sigma as a multiple of the training std
            q_bounds_in (list): [lower, upper] Q bounds for underdamped SHO terms
            require_mid (bool): if True, discard kernel combos without a mid-range term
            verbose / plot (bool): general output flags
            loop_verbose / loop_plot / loop_savefigs (bool): output flags within the model selection loop
            results_verbose / results_plot (bool): output flags for the best-model summary

        Out:
            best_gp (celerite2 GaussianProcess): the best-fit GP trained on the training set
        '''
      
        # Read the data into a "raw" df
        raw_df = pd.read_csv(datapath, sep=r'\s+', skip_blank_lines=True)

        # Prep the df adds the column names etc
        data_df = prepare_df(raw_df, add_prefix=add_prefix, relative=relative)

        # Split the data
        dirty_train_df, dirty_valid_df, test_df = split_df(data_df, train_split=train_split, valid_split=valid_split)

        # Clean the dataset for outliers
        train_df, valid_df, _ = clean_df(dirty_train_df, dirty_valid_df, tol=4, verbose=verbose, plot=plot)

        # Classify signal data and get priors
        classified_signal_data = find_classify_signals(dirty_train_df,
                                                plot_fitpeaks=plot, verbose_fitpeaks=verbose,
                                                plot_genpriors=plot, verbose_genpriors=verbose)
        
        rho_priors, rho_prior_bounds, rho_sources = get_priors(classified_signal_data,
                                                                        star_type=star_type,
                                                                        direct_bound_tol = direct_bound_tol, 
                                                                        sm2016_bound_tol = sm2016_bound_tol, 
                                                                        mean_bound_tol = mean_bound_tol,
                                                                        verbose=verbose)

        #-----Do the model training and comparison---------
        train_yerr = train_df['sind'] * error_percent / 100

        train_mean = train_df['sind'].mean()
        train_std  = train_df['sind'].std()

        # Define the prior combos to try (k, [[q_priors], [rho_priors]])
        prior_combos = {
                "1s":    {'k': 1, 'q_priors': None, 'cycle_keys': ['short']},
                "1m":    {'k': 1, 'q_priors': None, 'cycle_keys': ['mid']},
                "1l":    {'k': 1, 'q_priors': None, 'cycle_keys': ['long']},

                "2sm":   {'k': 2, 'q_priors': None, 'cycle_keys': ['short', 'mid']},
                "2ml":   {'k': 2, 'q_priors': None, 'cycle_keys': ['mid',   'long']},
                "2sl":   {'k': 2, 'q_priors': None, 'cycle_keys': ['short', 'long']},

                "3sml":  {'k': 3, 'q_priors': None, 'cycle_keys': ['short', 'mid', 'long']},

                "const": {'k': 1, 'q_priors': "overdamped", 'cycle_keys': ['mid']},
        }

        if require_mid:
                prior_combos = {key: val for key, val in prior_combos.items() if 'mid' in val['cycle_keys']}

        # Set up the loop
        best_combo = None
        best_NLPD  = np.inf
        best_gp = None
        best_params = None

        # Run the loop
        for combo_name, combo_info in prior_combos.items():
                k = combo_info['k']
                q_prior_type = combo_info['q_priors']
                cycle_keys = combo_info['cycle_keys']
                
                np.random.seed(SEED)

                #---- q priors----
                if q_prior_type == 'overdamped': # andles the constant regime
                        q_0s = [np.random.uniform(0, 0.5) for _ in range(k)] 
                else:
                        q_0s = [np.random.uniform(0.5, 1) for _ in range(k)]

                #---remaining priors----
                sigma_0s = [train_std / k for _ in range(k)]
                rho_0s = [rho_priors[cycle_key] for cycle_key in cycle_keys]

                initial_guess = np.concatenate([
                        np.log(sigma_0s),
                        np.log(rho_0s),
                        np.log(q_0s)
                ])

                #----q bounds-----
                if q_prior_type == 'overdamped':
                        q_bounds = [(-np.inf, np.log(0.5)) for _ in range(k)]
                else:
                        q_bounds = [(np.log(q_bounds_in[0]), np.log(q_bounds_in[1])) for _ in range(k)]

                sigma_upper = train_std * sigma_upper_mult
                sigma_bounds = [(np.log(1e-4), np.log(sigma_upper)) for _ in range(k)]
                rho_bounds = np.log([rho_prior_bounds[cycle_key] for cycle_key in cycle_keys])

                bounds = np.concatenate([sigma_bounds, rho_bounds, q_bounds])

                # Create the initial kernel for this iteration
                kernel = terms.SHOTerm(sigma=sigma_0s[0], rho=rho_0s[0], Q=q_0s[0])
                for k_idx in range(1, k):
                        kernel += terms.SHOTerm(sigma=sigma_0s[k_idx], rho=rho_0s[k_idx], Q=q_0s[k_idx])

                # Define and compute the GP
                gp = celerite2.GaussianProcess(kernel, mean=train_mean)
                gp.compute(train_df['day'], yerr=train_yerr)

                # Train the GP
                gp_results = minimize(
                        NLL, initial_guess,
                        args=(gp, k, train_df['sind'].to_numpy()),
                        method='L-BFGS-B',
                        bounds=bounds
                        )

                gp   = set_params(gp_results.x, k, gp)
                gp.recompute()
                best = np.exp(gp_results.x)

                if loop_verbose:
                        table = PrettyTable(["sigma", "Q", "rho (days)", "rho (years)"])
                        best_sigmas = best[0:k]
                        best_rhos   = best[k:2*k]
                        best_qs     = best[2*k:]
                        for k_idx in range(k):
                                table.add_row([f"{best_sigmas[k_idx]:.4f}", f"{best_qs[k_idx]:.2f}",
                                               f"{best_rhos[k_idx]:.2f}", f"{best_rhos[k_idx]/365:.2f}"])
                        print(gp_results.success, gp_results.message)
                        print(f"For k = {k}")
                        print(table)

                # Predict on validation set
                mu, cov = gp.predict(train_df['sind'], t=valid_df['day'], return_var=True)
                std = np.sqrt(cov)

                # Model comparison via NLPD
                y_valid    = valid_df['sind'].to_numpy()
                valid_yerr = valid_df['sind'] * error_percent / 100
                total_var  = cov + valid_yerr**2
                nlpd_per_point = 0.5 * np.log(2 * np.pi * total_var) + (y_valid - mu)**2 / (2 * total_var)
                mean_nlpd = nlpd_per_point.mean()

                if mean_nlpd < best_NLPD:
                        best_combo  = combo_name
                        best_NLPD   = mean_nlpd
                        best_gp     = gp
                        best_params = best

                if loop_plot:
                        loop_fig, loop_ax = plt.subplots(figsize=(20, 5))
                        loop_ax.scatter(train_df['year'], train_df['sind'], color='blue', label='Training', marker='x')
                        loop_ax.plot(valid_df['year'], valid_df['sind'], color='orange', label='Actual', alpha=0.5)
                        
                        loop_ax.set_title(star_name + " " + combo_name)

                        loop_ax.plot(valid_df['year'], mu, color='green', label='Predictions')
                        loop_ax.fill_between(valid_df['year'], mu - std, mu + std,
                                             color='green', alpha=0.2, label='Uncertainties')
                        
                        loop_ax.legend()

                        loop_ax.text(0.02, 0.05, f"NLPD = {mean_nlpd:.4f}",
                                transform=loop_ax.transAxes, fontsize=10, verticalalignment='bottom',
                                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
                        
                        if loop_savefigs:
                                loop_fig.savefig(f"./figs/gpr_{star_name}_{int(round(train_split * 100)):02d}_{combo_name}.png", dpi=100, bbox_inches='tight')
                                plt.close(loop_fig)

                        else:
                                plt.show()

                if loop_verbose:
                        print(f"Mean NLPD on validation set ({combo_name}): {mean_nlpd:.4f}")

        #-------Now we have the best model------

        # Predict on a dense regular grid for smooth plotting
        t_start = valid_df['day'].iloc[0]
        t_end   = valid_df['day'].iloc[-1] + result_plot_extra
        sampled_days = np.arange(t_start, t_end, result_plot_cadence)

        mu, cov = best_gp.predict(train_df['sind'], t=sampled_days, return_var=True)
        std = np.sqrt(cov)
        results = pd.DataFrame({
                'forecast': mu,
                'lower': mu - std,
                'upper': mu + std,
        }, index=sampled_days)

        # Convert sampled days to years for plotting
        day_ref = train_df['day'].iloc[0]
        year_ref = train_df['year'].iloc[0]
        sampled_years = year_ref + (sampled_days - day_ref) / 365.25

        if results_verbose: # Print best model params
                print(f"The best params were found for {best_combo} with NLPD of {best_NLPD}.")
                k = prior_combos[best_combo]['k']
                table = PrettyTable(["sigma", "Q", "rho (days)", "rho (years)"])
                best_sigmas = best_params[0:k]
                best_rhos = best_params[k:2*k]
                best_qs = best_params[2*k:]
                for k_idx in range(k):
                        table.add_row([f"{best_sigmas[k_idx]:.4f}", f"{best_qs[k_idx]:.2f}", f"{best_rhos[k_idx]:.2f}", f"{best_rhos[k_idx]/365:.2f}"])
                print(table)

        if results_plot: # Plot best model predictions
                if ax is None: # Enables passthrough
                        fig, ax = plt.subplots(figsize=(20, 5))
                ax.scatter(train_df['year'], train_df['sind'], color='blue', label='Training', marker='x')
                ax.scatter(valid_df['year'], valid_df['sind'], color='orange', label='Actual', alpha=0.5, marker = 'x')
                
                ax.set_title(f"Best model for {star_name} is {best_combo}")
                ax.plot(sampled_years, results['forecast'], color='green', label='Predictions')
                ax.fill_between(sampled_years, results['lower'], results['upper'],
                                color='green', alpha=0.2, label='Uncertainties')
                ax.legend()
                ax.text(0.02, 0.05, f"NLPD = {best_NLPD:.4f}",
                        transform=ax.transAxes, fontsize=10, verticalalignment='bottom',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        return best_gp