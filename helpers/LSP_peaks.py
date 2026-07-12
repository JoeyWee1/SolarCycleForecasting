import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

from astropy.time import Time
from astropy.timeseries import LombScargle

from scipy.signal import find_peaks
from scipy.optimize import curve_fit

from prettytable import PrettyTable

def check_aliases(freq, accepted_freqs, tol = 0.05):
    '''
    Checks whether a candidate frequency is an alias of any already-accepted frequency.
    Aliases are integer or simple-fraction multiples of accepted periods.
    They are noise and have to be removed.

    In:  
        - freq (float): the candidate frequency in days
        - accepted_freqs (list of floats): reference frequencies
        - tol (float): fractional tolerance 
        
    Out: 
        - not_alias (bool): True if the frequency is not an alias and should be considered for acceptance
        - aliased_period (float): the period for which the candidate period is an alias; None if not an alias
        - alial_multiplier (float): the concomitant multiple
        - delta_float (float): the tolerance by which this was identified as an alias
    '''
    accepted_periods = 1/np.array(accepted_freqs) # Check in period space
    for p in accepted_periods:
        for mult in [0.25, 1/3, 0.5, 1, 2, 3, 4]:
            pmult_freq = 1/(p*mult)
            delta = np.abs((freq - pmult_freq)/freq)
            if delta < tol: # The signal is an alias
                return False, p, mult, delta
    return True, None, None, None # None signatures to match other return

def check_windows(freq, tol = 0.05):
    '''
    Checks whether a candidate frequency matches a known observational window function
    (e.g. yearly, monthly, weekly, or lunar periods).

    In:
        - freq (float): the candidate frequency in days
        - tol (float): the fractional tolerance 

    Out: 
        - not_window (bool): True if the frequency does not match any window and should be kept
        - matched_window_days (float): The window is has been matched to; None of not a window
        - delta (float): The tolerance by which the matching is made; None if not a window

    '''
    # Times
    year = 365.25
    month = year /12
    week = 7
    windows = [year, year/2, year/3, year/4, month, 2*month, week, 2* week, 3 * week, 29.5] #29.5 is lunar
    for window in windows:
        window_freq = 1/np.array(window)
        delta = np.abs((freq-window_freq)/freq)
        if delta < tol: # Then we classify the signal as a window function
            return False, window, delta
    return True, None, None

# Helpers for performing the SNR checks
def gaussian(t, A, mu, std, c):
    '''
    Evaluates a Gaussian with a vertical offset.

    In:  
        - t (array of floats): the x axis
        - A (float): Gaussian amplitude
        - mu (float): the centre of the Gaussian in t-space
        - std (float): Gaussian standard deviation
        - c (float): y axis offset

    Out: 
        - Array of Gaussian values evaluated at t (array of floats)
    '''
    return A * np.exp(-0.5 * ((t - mu) / std)**2) + c

def find_snr(powers, freqs, peak_freq, peak_idx, peak_height, window = 150):
    '''
    Fits a Gaussian around a peak and computes SNR as peak amplitude over residual noise std.
    Falls back to (peak - median) / std if the Gaussian fit fails; these will be weaker than the 
    Gaussian fits because median will be higher than the residual noise std.

    In:  
        - powers (array of floats): the powers of the LSP
        - freqs (array of floats): the frequency x axis of the LSP
        - peak_freq (float): frequency of the peak considered; used as mu initial guess
        - peak_idx (int): array index of peak for window to be defined
        - peak_height (float): LSP power at the peak; used as amplitude initial guess
        - window (int): half-width of fitted region in indices

    Out:
        - snr (float): the calculated signal-to-noise ratio
        - popt (array of floats): the fit; None if fit failed
        - resid_lsp (array of floats): the residual LSPs; None of fit failed
    '''
    # Define a window around the peak to which to fit a Gaussian
    n = len(powers)
    left  = max(0, peak_idx - window)
    right = min(n, peak_idx + window + 1)
    window_freqs = freqs[left:right]
    window_powers = powers[left:right]
    minf = min(window_freqs)
    maxf = max(window_freqs)

    # Define initial guesses for the fit
    A0 = peak_height
    mu0 = peak_freq
    std0 = mu0 * 0.1 #FWHM is a bit overkill
    c0 = np.median(window_powers)
    p0 = [A0, mu0, std0, c0]

    # Try the curve fit
    try:
        popt, _ = curve_fit(gaussian, window_freqs, window_powers, # We don't care about the uncs at this step
                        p0 = p0,
                        bounds = ([0, minf, 0, 0], [np.inf, maxf, np.inf, np.inf]))
        resid_lsp = window_powers - gaussian(window_freqs, *popt) # The LSP after removing the gaussian
        noise_std = np.std(resid_lsp)
        snr = A0 / noise_std
        return snr, popt, resid_lsp
    except (RuntimeError, ValueError): # IF the fit does not work
        continuum = np.median(window_powers)
        noise = np.std(window_powers)
        return (A0 - continuum) / noise, None, None


def build_design_matrix(t, accepted_freqs):
    """
    Constructs the design matrix for a linear least-squares fit of multiple sine/cosine pairs
    with a constant offset. Each frequency contributes a cos and sin column.

    In:
        - t (array of floats): the time axis in days
        - accepted_freqs (list of floats): the accepted frequencies in cycles/day

    Out:
        - X (ndarray): design matrix of shape (n_obs, 1 + 2*n_freqs)
    """
    # Column of just ones for the offset component
    cols = [np.ones(len(t))]

    for f in accepted_freqs:
        cols.append(np.cos(2 * np.pi * f * t)) # Equivalent to 2 pi t / T
        cols.append(np.sin(2 * np.pi * f * t))
        
    return np.column_stack(cols) 

def simultaneous_fit(t, y, accepted_freqs):
    """
    Fits all accepted frequencies simultaneously via least squares and returns the residuals
    for the next LSP iteration.

    In:
        - t (array of floats): the time axis in days
        - y (array of floats): the S-index values
        - accepted_freqs (list of floats): the accepted frequencies in cycles per day

    Out:
        - residuals (array of floats): y minus the simultaneous fit
        - params (array of floats): fitted coefficients [offset, cos1, sin1, ...]
    """
    if not accepted_freqs:
        # If no frequencies yet, residuals are just the original data
        return y, np.array([np.mean(y)])

    # Build the matrix of params 
    X = build_design_matrix(t, accepted_freqs)
    
    # Solve the linear system X * params = y to fit
    params, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    
    # Calculate the model and the residuals
    fitted_model = X @ params
    residuals = y - fitted_model
    
    return residuals, params

def set_period_axes(ax, ylabel='Normalised Power'):
    '''
    Adds a secondary x-axis in years (top) to a matplotlib axis whose primary x-axis is in days (bottom).

    In:
        - ax (matplotlib Axes): axis with period in days on the x-axis
        - ylabel (str): label for the y-axis

    Out:
        - None
    '''
    ax2 = ax.secondary_xaxis('top', functions=(lambda d: d / 365.25, lambda y: y * 365.25))
    ax2.set_xlabel('Period (years)')
    ax.set_xlabel('Period (days)')
    ax.set_ylabel(ylabel)

# Now the function proper ----------

def fit_peaks(df,
            manual_freq = 'linear', period_range = [0.1, 100*365], n_periods = 100000,
            FAPs = [10,5,1,0.1], key_FAP_idx = -1,
            threshold = 5,
            plot = True, verbose = True):
    '''
    Iteratively finds significant LSP peaks, rejecting aliases, window functions, and low-SNR
    detections. At each iteration the accepted frequencies are subtracted via simultaneous fit
    before the next LSP is computed.

    In:
        - df (DataFrame): training dataframe with 'day' and 'sind' columns
        - manual_freq (str or None): 'linear', 'log', or None for astropy autopower
        - period_range (list): [min, max] period in days for the frequency grid
        - n_periods (int): number of frequency grid points
        - FAPs (list of floats): false alarm probability levels to be considered as percentages
        - key_FAP_idx (int): index into FAPs giving the detection threshold
        - threshold (float): minimum SNR required to accept a peak
        - plot (bool): whether to plot the LSP at each iteration
        - verbose (bool): whether to print accepted/rejected peak info

    Out:
        - accepted_peak_periods (array of floats): accepted periods in days
        - accepted_peak_heights (array of floats): LSP power at each accepted period
    '''
    accepted_peak_freqs = [] # We will work in frequency space for the analysis other than for the plotting. It is easier for the SNR fitting.
    accepted_peak_heights = [] # The concomitant power heights

    # Choice of the periods to be searched
    if manual_freq == 'linear':
        min_period = period_range[0]
        max_period = period_range[1]
        periods = np.linspace(min_period, max_period, n_periods)
        freqs = 1 / periods

    if manual_freq == 'log': # Use log regime to create uniform relative values 
        min_period = period_range[0]
        max_period = period_range[1]
        periods = np.logspace(np.log10(min_period), np.log10(max_period), n_periods)
        freqs = 1 / periods

    resids = [df['sind']] # The first residual is already defined

    # For each iteration store for plotting
    iter_powers = [] # Add the LSPs here
    iter_params = [] # Parameters of the simultaneous fit for each iteration
    iter_peak_freqs = [] # the full list of peaks found at each stage
    iter_invFAPs = [] # the list of powers corresponding to the FAPs
    iter_popts = [] # popts of the Gaussian fit
    iter_SNRs = []

    i = 0 # What index in resids we are presently processing
    t = df['day'] # Fixed time axis for al
    data_timeframe = t.iloc[-1] - t.iloc[0]
    max_detectable_period = 3 * data_timeframe #  2 sinusoids typically required to fit reliably

    # Now calculate the FAPs
    FAPs = np.array(FAPs)/100

    while True:
        # Take the LSP: powers for the freqs
        ls = LombScargle(t, resids[i])

        # Choice of which frequency range
        if manual_freq is None: # Use autopower
            freqs, powers = ls.autopower()
            periods = 1/freqs
        else:
            powers = ls.power(freqs)

        # Finds teh concomitant powers
        power_invFAPs = ls.false_alarm_level(FAPs, method = 'bootstrap') 
        key_FAP = float(power_invFAPs[key_FAP_idx])
        # Identify FAPs above the key FAP
        peak_idxs, properties = find_peaks(powers, height=key_FAP) # peaks is an array of the indices of the input arrays with the peaks
        peak_freqs = freqs[peak_idxs]
        peak_heights = properties['peak_heights']

        # Break if none
        if len(peak_freqs) == 0:
            break

        # Sort identified peaks by height
        sort_idx = np.argsort(peak_heights)[::-1]
        peak_freqs = peak_freqs[sort_idx]
        peak_heights = peak_heights[sort_idx] 
        peak_idxs = peak_idxs[sort_idx]

        found = False
        # Loop through peaks until one satisfies conditions or the end is reached
        for peak_freq, peak_height, peak_idx in zip(peak_freqs, peak_heights, peak_idxs):

            # Check if less than max_period
            if 1/peak_freq > max_detectable_period:
                if verbose: print(f"Peak at period {1/peak_freq:.2f} days is greater than maximum detectable period of {max_detectable_period} days.")
            # Less than maxperiod check if alias
            else:
                alias_cond, p, mult, delta = check_aliases(freq = peak_freq, accepted_freqs= accepted_peak_freqs)
                if alias_cond == False: # This is an alias; skip freq
                    if verbose: print(f"Peak at period {1/peak_freq:.2f} days was identified to be an alias of {p:.2f} days at mult {mult:.2f} by tolerance {delta:.2f}.")

                else: # Not an alias; now check window
                    window_cond, window, delta = check_windows(freq = peak_freq)
                    if window_cond == False: # This is a window function; skip freq
                        if verbose: print(f"Peak at period {1/peak_freq:.2f} days was identified to be a window of {window:.2f} days by tolerance {delta:.2f}.")

                    else: # Not a window; now check SNR
                        snr, popt, resid_lsr =  find_snr(powers, freqs, peak_freq, peak_idx, peak_height, window = 150)
                        if popt is None:
                            if verbose: print("SNR failed to fit Gaussian. Fell back to height vs average.")

                        if snr > threshold: # All checks passed. Append
                            accepted_peak_freqs.append(peak_freq) # Freq, height
                            accepted_peak_heights.append(peak_height)
                            found = True # indicates that we have added a frequency this time
                            if verbose: print(f"In iteration {i}, period at {1/peak_freq:.2f} days was accepted.")
                            break # stops searching freqs

                        else: # Go to nest frequency
                            if verbose: print(f"Period of {1/peak_freq:.2f} days has SNR of {snr} < {threshold}.")

        # if we have not found a frequency, this means that we have reached the FAP limit
        if found == False:
            if verbose: print(f"Iteration {i} did not find a frequency: terminating loop.")
            break

        else: # freq found now do the refitting for the next loop round
            resid, params = simultaneous_fit(df['day'], df['sind'], accepted_peak_freqs)
            resids.append(resid)
            iter_powers.append(powers) # The LSPs for plotting
            iter_params.append(params)
            iter_peak_freqs.append(peak_freqs)
            iter_invFAPs.append(power_invFAPs)
            iter_SNRs.append(snr)
            if popt is None:
                iter_popts.append(None)
            else:
                iter_popts.append(popt)
            i += 1 # Now the cycle repeats with the new set of resids

    periods = 1/freqs

    # Now we have a list of freqs: plot and return
    n_iter = len(resids)-1 # how many iterations we completed 
    if plot:
        fig, ax = plt.subplots(n_iter, 3, figsize=(10*3, 7*n_iter))
        ax = np.atleast_2d(ax)

        col_titles = ['LSP', 'Residual LSP', 'Fit & Residuals']
        for col, title in enumerate(col_titles):
            ax[0][col].set_title(title, fontsize=14, fontweight='bold')

        for row in range(n_iter):
            # ---- Col 0: LSP ----
            ax[row][0].set_xscale('log')
            ax[row][0].plot(periods, iter_powers[row], label = 'LS Periodogram')
            set_period_axes(ax[row][0])

            # plotting the FAPs
            fap_colors = ['#2ecc71', '#a8d44a', '#e67e22', '#e74c3c']
            power_invFAPs = iter_invFAPs[row]
            for power_invFAP, fap_color, fap_level in zip(power_invFAPs, fap_colors, FAPs):
                ax[row][0].axhline(power_invFAP, color=fap_color, label=f'FAP {fap_level * 100} %', lw=0.4)

            peak_periods_row = 1 / np.array(iter_peak_freqs[row])
            for j,peak_period in enumerate(peak_periods_row):
                ax[row][0].axvline(peak_period, color='purple', lw=0.4,
                                   label='Peak' if j == 0 else None)

            popt = iter_popts[row]
            if popt is not None:
                ax[row][0].plot(periods, gaussian(freqs, *popt), color='lime', label = "Gaussian fit")
            
            ax[row][0].legend()

            # ---- Col 1: Residual LSP (Gaussian subtracted) ----
            ax[row][1].set_xscale('log')
            ax[row][1].sharey(ax[row][0])
            ax[row][1].sharex(ax[row][0])
            set_period_axes(ax[row][1])
            if popt is not None:
                ax[row][1].plot(periods, iter_powers[row] - gaussian(freqs, *popt))

            # ---- Col 2: Previous residual, simultaneous fit, current residual vs time ----
            ax_t = ax[row][2]
            prev_resid = np.array(resids[row])
            curr_resid = np.array(resids[row + 1])
            fit_component = prev_resid - curr_resid  # sinusoidal component added this iteration

            t_vals = np.array(t)
            sort_idx = np.argsort(t_vals)

            ax_t.scatter(t_vals, prev_resid, s=2, color='orange', alpha=0.5, label='Previous residual', zorder=1)
            ax_t.scatter(t_vals, curr_resid, s=2, color='green', alpha=0.5, label='Residual', zorder=2)
            ax_t.plot(t_vals[sort_idx], fit_component[sort_idx], color='blue', lw=1, label='Simultaneous fit', zorder=3)

            ax_t.set_xlabel('Time (days)')
            ax_t.set_ylabel('S-index')
            ax_t.legend(fontsize=8, markerscale=3)

        plt.tight_layout()
        plt.show()
    
    accepted_peak_periods = 1/np.array(accepted_peak_freqs)
    if verbose:
        print("Accepted periods and their heights")
        table = PrettyTable(["Days", "Years", "Height", "SNR"])
        for accepted_peak_period, accepted_peak_height, snr in zip(accepted_peak_periods, accepted_peak_heights, iter_SNRs):
            table.add_row([f"{accepted_peak_period:.2f}", f"{accepted_peak_period/365:.2f}", f"{accepted_peak_height:.4f}", f"{snr:.2f}"])
        print(table)

    return accepted_peak_periods, accepted_peak_heights


