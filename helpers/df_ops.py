import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time

def prepare_df(data, add_prefix=False, relative = True):
    '''
    Takes the raw readcsv df and adds the labels to it.
    Relative makes the JD dates relative to the start.
    Day and year are still absolute.
    '''
    data = data.copy()
    data = data.iloc[:, :2].copy()  # guard against files with extra columns
    data.columns = ["JD", "sind"]
    # coerce non-numeric entries (e.g. stray header/comment rows) to NaN then drop
    data["JD"] = pd.to_numeric(data["JD"], errors='coerce')
    data["sind"] = pd.to_numeric(data["sind"], errors='coerce')
    data = data.dropna(subset=["JD", "sind"]).reset_index(drop=True)
    if add_prefix:
        data['JD'] = data['JD'] + 2400000.0
    time_obj = Time(data["JD"].to_numpy(dtype=float), format='jd', scale='tdb')
    if relative:
        data["JD"] = data["JD"] - data["JD"].iloc[0]    
    data["year"] = time_obj.jyear
    data["day"] = time_obj.jd
    data['datetime'] = time_obj.to_datetime(timezone=None)
    data = data.set_index('datetime')
    return data

def split_df(df, train_split=0.8, valid_split=0.19):
    '''
    Does the test-train-split of the df.
    '''
    n = len(df)
    train_idx = round(train_split * n)
    valid_idx = round((train_split + valid_split) * n)
    return df.iloc[:train_idx].copy(), df.iloc[train_idx:valid_idx].copy(), df.iloc[valid_idx:].copy()


def clean_df(train_df, valid_df, tol = 3, plot = True, verbose = True):
    '''
    This uses a median absolute deviance filter to remove any crazy outliers in the training data.
    Uses the same limit to remove ourliers in the validation data

    Returns a df with the outliers removed.
    If plot, plots the removed values.
    If verbose, states number of removed values.
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
    days = df['day'].values
    keep = np.zeros(len(days), dtype=bool)
    keep[0] = True
    last = days[0]
    for i in range(1, len(days)):
        if days[i] - last >= minimum_gap:
            keep[i] = True
            last = days[i]
    return df[keep].reset_index(drop=True)