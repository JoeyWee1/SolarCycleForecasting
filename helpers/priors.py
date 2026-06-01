import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

from astropy.time import Time
from astropy.timeseries import LombScargle

from scipy.signal import find_peaks
from scipy.optimize import curve_fit

from prettytable import PrettyTable

from helpers.LSP_peaks import set_period_axes, fit_peaks




def generate_priors(df,
        manual_freq = 'linear', period_range = [0.1, 100*365], n_periods = 100000,  # this is all the fit_peaks stuff
        FAPs = [10,5,1,0.1], key_FAP_idx = -1, 
        threshold = 5,
        plot_fitpeaks = True, verbose_fitpeaks = True,
        s_lim = 200, m_lim = 20 * 365, 
        plot_genpriors = True, verbose_genpriors = True,
        ):
    '''
    Generates the prior guesses for each of the three ranges for the spectrum kernel.

    Takes in the DF, calculates the LSP, identifies the peaks and the number thereof.
    It uses simple heuristics to classify them into long, mid, and short ranges.

    Params
    plot: plots LSP if True
    min_period, max_period in years
    n_periods to set the resolution
    s_lim = 150 days is approx what SM2016 says so 200 is safe
    m_lim = 14 years so do 20 * 365 to be safe
    
    Returns
    The peak periods and their concomitant ranges.
    '''
    peak_periods, peak_heights = fit_peaks(df, 
                    manual_freq = manual_freq, period_range = period_range, n_periods = n_periods,  # this is all the fit_peaks stuff
                    FAPs = FAPs, key_FAP_idx = key_FAP_idx, 
                    threshold = threshold,
                    plot = plot_fitpeaks, verbose = verbose_fitpeaks)

    # Now take the highest peak in each range
    priors = {}

    # Take a LS to compare the powers on the same scale
    t = df['day']
    y = df['sind']

    # ls = LombScargle(t, y)
    
    # peak_freqs = 1/peak_periods
    # peak_powers = ls.power(peak_freqs)

    ls = LombScargle(t, y)

    if manual_freq == 'linear':
        periods = np.linspace(period_range[0], period_range[1], n_periods)
    elif manual_freq == 'log':
        periods = np.logspace(np.log10(period_range[0]), np.log10(period_range[1]), n_periods)
    else:
        freqs, powers = ls.autopower()
        periods = 1 / freqs

    if manual_freq is not None:
        freqs = 1 / periods
        powers = ls.power(freqs)

    peak_powers = ls.power(1 / np.array(peak_periods))

    # Peak data
    peak_data = list(zip(peak_periods, peak_powers))

    peaks_s = [(p, pw) for p, pw in peak_data if p <= s_lim]
    peaks_m = [(p, pw) for p, pw in peak_data if s_lim < p <= m_lim]
    peaks_l = [(p, pw) for p, pw in peak_data if m_lim < p]

    if peaks_s:
            priors['short'] = max(peaks_s, key=lambda x: x[1])[0]
    if peaks_m:
            priors['mid'] = max(peaks_m, key=lambda x: x[1])[0]
    if peaks_l:
            priors['long'] = max(peaks_l, key=lambda x: x[1])[0]

    if plot_genpriors: # Plot first order LSP log-log and indicate the priors
        colors = {'short': 'red', 'mid': 'green', 'long': 'orange'}

        fig, ax = plt.subplots(figsize=(20,5))
        ax.set_xscale('log')
        ax.set_yscale('log')
        set_period_axes(ax)
       
        ax.plot(periods, powers, label = "LS Periodogram")
        
        for cycle_type, peak_period in priors.items():
                ax.axvline(peak_period, color = colors[cycle_type], label = f'Cycle type: {cycle_type}')
        
        ax.legend()
        plt.show()
    
    if verbose_genpriors:
        print("Priors")
        table = PrettyTable(["Cycle Type", "Prior Days", "Prior Years"])
        if priors.get('short'):
            table.add_row(["Short", f"{priors.get('short'):.2f}", f"{priors.get('short')/365:.2f}"])
        if priors.get('mid'):
            table.add_row(["Mid", f"{priors.get('mid'):.2f}", f"{priors.get('mid')/365:.2f}"])
        if priors.get('long'):
            table.add_row(["Long", f"{priors.get('long'):.2f}", f"{priors.get('long')/365:.2f}"])
        print(table)
    
    return priors