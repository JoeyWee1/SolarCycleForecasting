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

#------------------------------

def SM2016_intercepts(a):
    '''
    Gets the y intercept of the SM2016 formula.
    Uses the mean values from Table 1 and Table 2 in the paper.
    Everything in FGK order
    '''
    mu_Pcycs = np.array([9.5, 6.7, 8.5]) # years
    mu_Pcycs *= 365 # now in days

    mu_Prots = np.array([8.6, 19.6, 27.4])

    c = np.log10(mu_Pcycs) - (1-a)*np.log10(mu_Prots)
    return {'F': c[0], 'G': c[1], 'K': c[2]}

def SM2016_s_to_m(P_rot, star_type, a = 0.89):
    '''
    ONLY VALID FOR FGK STARS
    This takes in the short period and return the mid period.
    Calculated using the SM2016 relation.
    Slope a = 0.89 pm 0.05 betwen x-> log(1/P_rot) and y-> log(P_cyc/P_rot)
    '''
    c = SM2016_intercepts(a)[star_type]
    log10_Pcyc = (1-a)* np.log10(P_rot) + c
    P_cyc = 10 ** log10_Pcyc
    return P_cyc

def SM2016_m_to_s(P_cyc, star_type, a = 0.89):
    '''
    ONLY VALID FOR FGK STARS
    This takes in the mid period and returns the short period.
    Calculated using hte SM2016 relation.
    Slope a = 0.89 pm 0.05 betwen x-> log(1/P_rot) and y-> log(P_cyc/P_rot)
    '''
    c = SM2016_intercepts(a)[star_type]
    log10_Prot = (1/(1-a))*(np.log10(P_cyc)-c)
    P_rot = 10 ** log10_Prot
    return P_rot

#------------------------------

def find_classify_signals(df,
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
        print("Classified Signals")
        table = PrettyTable(["Cycle Type", "Prior Days", "Prior Years"])
        if priors.get('short'):
            table.add_row(["Short", f"{priors.get('short'):.2f}", f"{priors.get('short')/365:.2f}"])
        if priors.get('mid'):
            table.add_row(["Mid", f"{priors.get('mid'):.2f}", f"{priors.get('mid')/365:.2f}"])
        if priors.get('long'):
            table.add_row(["Long", f"{priors.get('long'):.2f}", f"{priors.get('long')/365:.2f}"])
        print(table)
    
    return priors # Classified signal data, not priors. This goes into the get priors func. 


def get_priors(classified_signal_data, star_type, verbose = True):
    '''
    This takes in the input of the find_classify_signals function
    and outputs the three priors. One in each range.count
    Fills in missing ones using the SM2016 relationship.
    Fills in missing long-term priors with 100 year guess.
    '''
    mu_Pmids = { # THESE COME FROM SM2016
    'F':  {'mean_yr': 9.5,  'mean_days': 9.5  * 365},
    'G':  {'mean_yr': 6.7,  'mean_days': 6.7  * 365},
    'K':  {'mean_yr': 8.5,  'mean_days': 8.5  * 365},
    'ME': {'mean_yr': 6.0,  'mean_days': 6.0  * 365},
    'MM': {'mean_yr': 7.1,  'mean_days': 7.1  * 365},
    }  # P_cyc

    mu_Pshorts = {
        'F':  {'mean_days': 8.6},
        'G':  {'mean_days': 19.6},
        'K':  {'mean_days': 27.4},
        'ME': {'mean_days': 36.2},
        'MM': {'mean_days': 85.4},
    }  # P_rot

    # If has all three then return all three
    short = classified_signal_data.get('short', np.nan)
    mid = classified_signal_data.get('mid', np.nan)
    long = classified_signal_data.get('long', np.nan)

    # Use detected if available, otherwise use SM2016 if available, else use mean for that star type
    def _resolve(direct, derive, default):
      if not np.isnan(direct):
          return direct, 'found'
      derived = derive()
      if derived is not None:
          return derived, 'SM2016'
      return default, 'mean'

    short_val, short_src = _resolve(
        short,
        lambda: SM2016_m_to_s(mid, star_type) if not np.isnan(mid) else None,
        mu_Pshorts[star_type]['mean_days']
    )
    mid_val, mid_src = _resolve(
        mid,
        lambda: SM2016_s_to_m(short, star_type) if not np.isnan(short) else None,
        mu_Pmids[star_type]['mean_days']
    )
    long_val, long_src = _resolve(long, lambda: None, 100 * 365)

    priors = {'short': short_val, 'mid': mid_val, 'long': long_val}

    if verbose:
        table = PrettyTable(["Cycle Type", "Prior Days", "Prior Years", "Source"])
        table.add_row(["Short", f"{priors.get('short'):.2f}", f"{priors.get('short')/365:.2f}", short_src])
        table.add_row(["Mid", f"{priors.get('mid'):.2f}", f"{priors.get('mid')/365:.2f}", mid_src])
        table.add_row(["Long", f"{priors.get('long'):.2f}", f"{priors.get('long')/365:.2f}", long_src])
        print(table)

    return priors


