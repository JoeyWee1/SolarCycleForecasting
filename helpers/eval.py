import numpy as np
import matplotlib.pyplot as plt
import emcee
from scipy.stats import gaussian_kde
from prettytable import PrettyTable


def check_constant(predictions, MAD, tol=2, percentile=84):
    '''
    Flags whether GP predictions are near-constant, indicating the model has not found a cycle.
    Uses the chosen percentile of peak-to-trough amplitude across MCMC draws, relative to the MAD.

    In:
        - predictions (ndarray): shape (n_samples, n_times) array of MCMC prediction draws
        - MAD (float): median absolute deviation from clean_df, used as a noise reference
        - tol (float): amplitude/MAD ratio below which predictions are flagged as constant
        - percentile (float): which percentile of per-draw amplitudes to use

    Out:
        - is_constant (bool): True if predictions are near-constant
    '''
    amps = predictions.max(axis=1) - predictions.min(axis=1)
    return np.percentile(amps, percentile) / MAD < tol


def best_in_x(predictions, t, lookahead_years, t_start=None):
    '''
    Finds the predicted activity minimum within a lookahead window starting at t_start.

    In:
        - predictions (ndarray): shape (n_samples, n_times) array of MCMC prediction draws
        - t (array of floats): time axis in days, matching axis 1 of predictions
        - lookahead_years (float): length of the lookahead window in years
        - t_start (float): start of the window in days; defaults to t[0]

    Out:
        - median_min_day (float): median predicted time of minimum in days
        - bounds (tuple): (lb_day, ub_day) 16th and 84th percentile bounds in days
    '''
    t_start = t_start if t_start is not None else t[0]
    t_end = t_start + lookahead_years * 365
    if t_end > t[-1]: # Fail gracefully if t_end before the last prediction
        t_end = t[-1]
    mask = (t >= t_start) & (t < t_end) # Onlt the values within the lookahead window targeted
    window_t = t[mask]
    min_t = window_t[np.argmin(predictions[:, mask], axis=1)]
    return np.median(min_t), tuple(np.percentile(min_t, [16, 84]))


def fit_Fourier(raw_data_df, MAD, med_ref, rho_priors, test_df,
                error_percent, n_walkers, subsample=500):
    '''
    Fits a 3-harmonic Fourier model via MCMC to serve as the ground-truth activity minimum reference.
    Applies an exponential time-weighting that downweights old observations in the training set.

    In:
        - raw_data_df (DataFrame): full data with 'day', 'year', and 'sind' columns
        - MAD (float): median absolute deviation threshold for restricting to the minimum region
        - med_ref (float): median S-index of the training set, used as the MAD reference
        - rho_priors (dict): must contain 'mid' key giving the mid-cycle period in days
        - test_df (DataFrame): used to set the exponential weighting reference date
        - error_percent (float): assumed percentage measurement error on S-index
        - n_walkers (int): number of emcee walkers
        - subsample (int): number of MCMC draws to retain for output

    Out:
        - model_preds (ndarray): shape (subsample, 2000) array of Fourier model samples
        - mean_pred (ndarray): shape (2000,) posterior mean
        - lb (ndarray): shape (2000,) 16th percentile
        - ub (ndarray): shape (2000,) 84th percentile
        - t_plot_year (ndarray): shape (2000,) year axis for the above arrays
    '''
    # Clean the dataset with MAD 
    fourier_df = raw_data_df[(raw_data_df['sind'] - med_ref).abs() < MAD].reset_index(drop=True)
    raw_std = np.std(fourier_df['sind'])

    # Define the sampling bounds
    log_AB_bounds = np.log([[1e-4, 5 * raw_std] for _ in range(6)])
    log_T_bounds = np.log([[1e-4, raw_data_df['day'].iloc[-1] - raw_data_df['day'].iloc[0]]])
    c_bounds = [[fourier_df['sind'].min(), fourier_df['sind'].max()]]
    bounds = np.concatenate([log_AB_bounds, log_T_bounds, c_bounds])

    # Reference dates
    t0_ref_day  = raw_data_df['day'].iloc[0]
    t0_ref_year = raw_data_df['year'].iloc[0]

    # Date helper based on these values
    def to_year(day): 
        return t0_ref_year + (day - t0_ref_day) / 365.25

    # Posterior function was analytically derived
    def lnPost_Fourier(params, t, y, sigma, weights, bounds):
        if np.any(params < bounds[:, 0]) or np.any(params > bounds[:, 1]):
            return -np.inf
        A1, A2, A3 = np.exp(params[0:3])
        B1, B2, B3 = np.exp(params[3:6])
        T = np.exp(params[6]); c = params[7]
        model  = c + A1*np.sin(2*np.pi*t/T) + B1*np.cos(2*np.pi*t/T)
        model += A2*np.sin(4*np.pi*t/T)     + B2*np.cos(4*np.pi*t/T)
        model += A3*np.sin(6*np.pi*t/T)     + B3*np.cos(6*np.pi*t/T)
        return np.sum(weights * (-np.log(sigma) - 0.5*np.log(2*np.pi) - 0.5*((y - model)/sigma)**2))

    # Initial guesses
    ig_AB = [(fourier_df['sind'].max() - fourier_df['sind'].min()) / 6] * 6
    ig = np.concatenate([np.log(ig_AB), np.log([rho_priors['mid']]), [np.median(fourier_df['sind'])]])
    wsc = ig + 1e-4 * np.random.randn(n_walkers, len(ig)) # Walker starting positions are stochastic around

    t0_day = test_df['day'].iloc[0]
    t_arr = fourier_df['day'].to_numpy()
    y_arr = fourier_df['sind'].to_numpy()
    sigma = y_arr * error_percent / 100
    weights = np.where(t_arr < t0_day, np.exp(-(t0_day - t_arr) / rho_priors['mid']), 1.0)

    # Sample
    sampler = emcee.EnsembleSampler(n_walkers, len(ig), lnPost_Fourier,
                                    args=(t_arr, y_arr, sigma, weights, bounds))
    sampler.run_mcmc(wsc, nsteps=10000, progress=False)
    flat = sampler.get_chain(discard=5000, flat=True) # Remove burn in

    # The time axis for plotting is used to generate the predictions
    t_plot = np.linspace(raw_data_df['day'].min(), raw_data_df['day'].max(), 2000)
    t_plot_year = to_year(t_plot)
    step = max(1, len(flat) // subsample)

    # Make the predictions
    model_preds = []
    for s in flat[::step]:
        A1, A2, A3 = np.exp(s[0:3]); B1, B2, B3 = np.exp(s[3:6])
        T = np.exp(s[6]); c = s[7]
        m  = c + A1*np.sin(2*np.pi*t_plot/T) + B1*np.cos(2*np.pi*t_plot/T)
        m += A2*np.sin(4*np.pi*t_plot/T)     + B2*np.cos(4*np.pi*t_plot/T)
        m += A3*np.sin(6*np.pi*t_plot/T)     + B3*np.cos(6*np.pi*t_plot/T)
        model_preds.append(m)

    # Posterior averaging and returning
    model_preds = np.array(model_preds)
    return (model_preds, model_preds.mean(0),
            np.percentile(model_preds, 16, 0), np.percentile(model_preds, 84, 0),
            t_plot_year)


def truth_in_x(model_preds, t_plot_year, t0_year, lookahead_years):
    '''
    Extracts the Fourier ground-truth activity minimum within a lookahead window.

    In:
        - model_preds (ndarray): shape (n_samples, n_times) Fourier model draws
        - t_plot_year (array of floats): year axis matching axis 1 of model_preds
        - t0_year (float): start of the lookahead window in years
        - lookahead_years (float): length of the window in years

    Out:
        - median_min_year (float): median predicted time of minimum in years
        - bounds (tuple): (lb_year, ub_year) 16th and 84th percentile bounds in years
    '''
    mask = (t_plot_year >= t0_year) & (t_plot_year <= t0_year + lookahead_years)
    window_t  = t_plot_year[mask]
    min_years = window_t[np.argmin(model_preds[:, mask], axis=1)]
    lb, ub = np.percentile(min_years, [16, 84])
    return float(np.median(min_years)), (float(lb), float(ub))


def plot_return_errors(df, max_lookahead: int = 5):
    '''
    Plots the absolute-error KDE for each integer lookahead year and prints the posterior mean error.

    In:
        - df (DataFrame): results with 'lookahead_years' and 'error' columns
        - max_lookahead (int): maximum lookahead year to include

    Out:
        - None
    '''
    # Extract the lookaheads
    integer_lookaheads = sorted(df[(df['lookahead_years'] % 1 == 0) & (df['lookahead_years'] <= max_lookahead)]['lookahead_years'].unique())
    
    fig, ax = plt.subplots(figsize=(10, 5), dpi=200)
    colors = plt.cm.viridis(np.linspace(0, 1, len(integer_lookaheads)))
    post_means = []

    for color, la in zip(colors, integer_lookaheads): # Plot each lookahead's KDE
        d = df[df['lookahead_years'] == la]['error'].dropna()
        xs = np.linspace(0, la, 300)
        kde_vals = gaussian_kde(d)(xs) + gaussian_kde(d)(-xs)
        kde_vals /= np.trapezoid(kde_vals, xs) #Normalise
        post_mean = np.trapezoid(xs * kde_vals, xs)
        post_means.append([la, post_mean])
        ax.plot(xs, kde_vals, color=color, lw=1.5, label=f'{int(la)} yr')
        ax.axvline(post_mean, color=color, lw=1, ls='--')

    ax.set_xlabel('Error (years)')
    ax.set_ylabel('Probability Density')
    ax.legend(title='Lookahead', bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=12)
    plt.tight_layout()
    plt.show()

    # Print results
    t = PrettyTable(['Lookahead (yr)', 'Post Mean (yr)'])
    t.float_format = '0.3'
    for la, pm in post_means:
        t.add_row([int(la), pm])
    print(t)

def plot_cadence_analysis(df, max_lookahead: int = 5, savefig = None):
    '''
    Plots absolute-error KDEs per lookahead year, with one curve per sampling cadence.

    In:
        - df (DataFrame): results with 'lookahead_years', 'sampling_rate_days', and 'error' columns
        - max_lookahead (int): maximum lookahead year to include
        - savefig (str or None): file path to save the figure; displays if None

    Out:
        - None
    '''
    # Get the first max_years of lookaheads
    integer_lookaheads = sorted(df[(df['lookahead_years'] % 1 == 0) & (df['lookahead_years'] <= max_years)]['lookahead_years'].unique())
    sampling_rates_sorted = sorted(df['sampling_rate_days'].unique())

    # Defining the plotting grid
    n_la = len(integer_lookaheads)
    ncols = 3
    nrows = int(np.ceil(n_la / ncols))

    # Defining the colour scale for each 
    rate_colours = {r: c for r, c in zip(sampling_rates_sorted, plt.cm.plasma(np.linspace(0.1, 0.9, len(sampling_rates_sorted))))}

    # Create axes
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), sharey=False)
    axes = axes.flatten()

    # Loop through lookaheads
    for ax_idx, la in enumerate(integer_lookaheads):
        ax = axes[ax_idx]
        # Loop through the sampling rates to plot for it
        for rate in sampling_rates_sorted:
            d = df[(df['lookahead_years'] == la) & (df['sampling_rate_days'] == rate)]['error'].dropna() # Drop errors
            xs = np.linspace(0, la, 300)
            kde_vals = gaussian_kde(d)(xs) + gaussian_kde(d)(-xs) # Absolute error
            kde_vals /= np.trapezoid(kde_vals, xs) # Normalise to proper PDF
            ax.plot(xs, kde_vals, color=rate_colours[rate], lw=1.5, label=f'{rate:.0f} d')

        ax.set_title(f'Lookahead = {int(la)} yr')
        ax.set_xlabel('Error Magnitude (years)')
        ax.set_ylabel('Density')

    # If there are too many grid axes
    for ax in axes[n_la:]:
        ax.set_visible(False)

    axes[n_la - 1].legend(title='Cadence', bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8)

    fig.tight_layout()
    if savefig:
        fig.savefig(savefig)
    plt.show()