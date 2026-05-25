from astropy.timeseries import LombScargle
from scipy.signal import find_peaks
from bisect import bisect_right

def identify_peaks(t=None,y=None,df = None, plot = True, min_period = 0.01, max_period = 200, n_periods = 10000, normalise = True, max_peaks = 3):
    '''
    Identifies the peaks available to be used as priors in a GPR fitting.
    Takes in the DF, calculates the LSP, identifies the peaks and the number thereof.
    It uses simple heuristics to classify them into long, mid, and short ranges.

    Params
    plot: plots LSP if True
    min_period, max_period in years
    n_periods to set the resolutio 


    Returns
    The peak periods and their concomitant ranges.
    '''
    # Define freq range
    periods = np.linspace(min_period, max_period, n_periods)
    freq = 1 / periods

    if df is not None:
        ls = LombScargle(df['year'], df['sind'])

    elif t is not None and y is not None:
        ls = LombScargle(t, y)

    else:
        raise ValueError("Please define either the df, or t and y")
    
    # Get the powers for the corresponding frequencies
    powers = ls.power(freq)

    # Normalise the powers to frac of max for interpretability
    if normalise:
        max_power = max(powers)
        powers /= max_power

    # Define the noise floor against which to compare the peaks
    noise = np.percentile(powers, 75)
    
    # Find the Lomb-Scargle peaks
    peak_idxs, properties = find_peaks(powers, height = 0., prominence=  noise) # peaks is an array of the indices of the input arrays with the peaks
    peak_periods = periods[peak_idxs]
    peak_powers = powers[peak_idxs] # the concomitant powers

    # Counts the number of peaks
    n_peaks =  len(peak_periods)
    if n_peaks > max_peaks: # must choose the top three peaks
        # more than three peaks must be noise
        top_idx = np.argsort(peak_powers)[-1 * max_peaks:] # takes top three
        peak_periods = peak_periods[top_idx]
        peak_powers = peak_powers[top_idx]
        n_peaks = max_peaks

    ## Identify which cycle range the periods are in
    peak_info = {} # the dictionary to return with peaks in "range": length
    # 1 significant peak
    if n_peaks == 1:
        peak_period = peak_periods[0]
        if peak_period <= 1: # if less than one year it is the short (rotation) cycle
            peak_info['short'] = peak_period
        elif peak_period <= 50: # this is probably the mid (stellar/Schwabe) cycle
            peak_info['mid'] = peak_period
        else: # highly unlikely to be physical
            pass # returns empty peak_info to indicate that we should randomise priors and flag a warning
    
    # 2 significant peaks
    elif n_peaks == 2:
        shorter_period = min(peak_periods)
        longer_period = max(peak_periods)

        # best case: they fit in the short and mid ranges
        if 0 <= shorter_period <= 1 and  1 < longer_period <= 50:
            peak_info['short'] = shorter_period
            peak_info['mid'] = longer_period
        # elif one short one long
        elif 0 <= shorter_period <= 1 and 50 < longer_period:
            peak_info['short'] = shorter_period
            peak_info['long'] = longer_period
        #elif one mid one long
        elif 1 < shorter_period <= 50 and 50 < longer_period:
            peak_info['mid'] = shorter_period
            peak_info['long'] = longer_period
        # elif they are both in the short cycle range; unlikely contingency
        elif 0 <= shorter_period <= 1 and  0 <= longer_period <= 1: 
            peak_info['short'] = np.mean([shorter_period, longer_period])
        # elif they are both in the mid cycle range; unlikely contingency
        elif 1 <= shorter_period <= 50 and  1 <= longer_period <= 50: 
            peak_info['mid'] = np.mean([shorter_period, longer_period])
        # else they would both be in the Gleissberg/long regime: unphysical return empty
        else:    
            pass

    # 3 significant peaks
    else: 
        peak_periods.sort()
        peak_powers_sorted = peak_powers  # need to carry powers through the sort

        peak_data = sorted(zip(peak_periods, peak_powers), key=lambda x: x[0])
        peak_periods_s = [(p, pw) for p, pw in peak_data if p <= 1]
        peak_periods_m = [(p, pw) for p, pw in peak_data if 1 < p <= 50]
        peak_periods_l = [(p, pw) for p, pw in peak_data if p > 50]

        if peak_periods_s:
            peak_info['short'] = max(peak_periods_s, key=lambda x: x[1])[0]
        if peak_periods_m:
            peak_info['mid'] = max(peak_periods_m, key=lambda x: x[1])[0]
        if peak_periods_l:
            peak_info['long'] = max(peak_periods_l, key=lambda x: x[1])[0]

        # s_idx = bisect_right(peak_periods, 1)
        # m_idx = bisect_right(peak_periods, 50)
        # periods_s = peak_periods[: s_idx]
        # periods_m = peak_periods[s_idx:m_idx]
        # periods_l = peak_periods[m_idx:]
        # mean_s = np.mean(periods_s) if len(periods_s) != 0 else np.nan
        # mean_m = np.mean(periods_m) if len(periods_m) != 0 else np.nan
        # mean_l = np.mean(periods_l) if len(periods_l) != 0 else np.nan
        # if not np.isnan(mean_s):
        #     peak_info['short'] = mean_s
        # if not np.isnan(mean_m):
        #     peak_info['mid'] = mean_m
        # if not np.isnan(mean_l):
        #     peak_info['long'] = mean_l
    

    ## Plot
    if plot:
        colors = {
            'short': 'red',
            'mid': 'green',
            'long': 'orange',
        }
        fig, ax = plt.subplots(1, figsize = (20,5))
        ax.scatter(periods,powers, marker = 'x', color = 'blue', label = "Periodogram")
        for cycle_type, peak_period in peak_info.items():
            ax.axvline(peak_period, color = colors[cycle_type], label = f"Cycle type: {cycle_type}")
        ax.axhline(noise, label = "Noise level", color = 'purple')
        ax.legend()
        plt.show()

    return peak_info

    