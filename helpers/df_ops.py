import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time

def prepare_df(data, add_prefix=False, relative = True):
    '''
    Parses a raw two-column (JD, S-index) DataFrame and adds time columns.

    In:
        - data (DataFrame): raw two-column data as read from file
        - add_prefix (bool): add 2400000 to JD if the file uses truncated Julian dates
        - relative (bool): shift JD so it starts at 0

    Out:
        - data (DataFrame): columns [JD, sind, year, day, datetime], indexed by datetime
    '''
    data = data.copy()
    data = data.iloc[:, :2].copy()  # guard against files with extra columns
    data.columns = ["JD", "sind"]

    # drop non-numeric entries
    data["JD"] = pd.to_numeric(data["JD"], errors='coerce')
    data["sind"] = pd.to_numeric(data["sind"], errors='coerce')
    data = data.dropna(subset=["JD", "sind"]).reset_index(drop=True)

    if add_prefix:
        data['JD'] = data['JD'] + 2400000.0
    time_obj = Time(data["JD"].to_numpy(dtype=float), format='jd', scale='tdb')
    if relative:
        data["JD"] = data["JD"] - data["JD"].iloc[0]    

    # Sets the stuff as asked
    data["year"] = time_obj.jyear
    data["day"] = time_obj.jd
    data['datetime'] = time_obj.to_datetime(timezone=None)
    data = data.set_index('datetime')
    return data

def split_df(df, train_split=0.8, valid_split=0.19):
    '''
    Splits a time-ordered DataFrame into train, validation, and test sets by row fractions.

    In:
        - df (DataFrame): time-ordered data
        - train_split (float): fraction of rows for training
        - valid_split (float): fraction of rows for validation; remainder goes to test

    Out:
        - train_df (DataFrame): first train_split fraction of rows
        - valid_df (DataFrame): next valid_split fraction of rows
        - test_df (DataFrame): remaining rows
    '''
    n = len(df)
    train_idx = round(train_split * n)
    valid_idx = round((train_split + valid_split) * n)
    return df.iloc[:train_idx].copy(), df.iloc[train_idx:valid_idx].copy(), df.iloc[valid_idx:].copy()


def clean_df(train_df, valid_df, tol = 3, plot = True, verbose = True):
    '''
    Removes outliers using a MAD filter computed on the training set.
    The same threshold is applied to the validation set to keep them consistent.

    In:
        - train_df (DataFrame): training data with 'sind' column
        - valid_df (DataFrame): validation data with 'sind' column
        - tol (float): number of MADs from the median allowed before a point is removed
        - plot (bool): whether to plot the removed points
        - verbose (bool): whether to print counts of removed points

    Out:
        - cleaned_train_df (DataFrame): training data with outliers removed
        - cleaned_valid_df (DataFrame): validation data with outliers removed
        - MAD (float): the threshold value (tol * median absolute deviation)
    '''
    l1 = len(train_df)
    train_df = train_df.dropna()
    l2 = len(train_df)

    if verbose:
        print(f'Removed {l1 - l2} datapoints which had NaN.')

    med = train_df['sind'].median()
    mad = (train_df['sind'] - med).abs().median()

    cleaned_train_df = train_df[(train_df['sind'] - med).abs() < tol * mad]
    cleaned_valid_df = valid_df[(valid_df['sind'] - med).abs() < tol * mad]


    if verbose:
        l1 = len(train_df)
        l2 = len(cleaned_train_df)
        print(f"Removed {l1 - l2} datapoints that were deemed ourliers by tolerance in sind of {tol} times MAD.")
    
    if plot:
        fig, ax =plt.subplots(figsize=(20,5))
        ax.scatter(train_df['day'], train_df['sind'], 
           s=10, label='Removed points', alpha=0.5, color='red')
        ax.scatter(cleaned_train_df['day'], cleaned_train_df['sind'], 
                s=10, label='Remaining points', alpha=0.8, color='blue')
        ax.axhline((tol * mad)+med, label = "Threshold", color = 'orange', linestyle = "--")
        ax.legend()
        ax.set_xlabel("Days")
        ax.set_ylabel("SInd")
        plt.tight_layout()
        plt.show()
        
    return cleaned_train_df, cleaned_valid_df, tol * mad

def downsample_min_gap(df, minimum_gap):
    '''
    Downsamples a DataFrame by enforcing a minimum gap between successive observations.
    Scans chronologically and retains a point only if it is at least minimum_gap days after the last kept point.

    In:
        - df (DataFrame): data with 'day' column, sorted chronologically
        - minimum_gap (float): minimum separation in days between retained observations

    Out:
        - df (DataFrame): downsampled data with reset index
    '''
    days = df['day'].values
    keep = np.zeros(len(days), dtype=bool) # Define a mask
    keep[0] = True
    last = days[0]
    for i in range(1, len(days)):
        if days[i] - last >= minimum_gap:
            keep[i] = True
            last = days[i]
    return df[keep].reset_index(drop=True)