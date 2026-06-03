import pandas as pd
import matplotlib.pyplot as plt
from astropy.time import Time

def prepare_df(data, add_prefix=False):
    '''
    Takes the raw readcsv df and adds the labels to it.
    '''
    data = data.copy()
    data.columns = ["JD", "sind"]
    if add_prefix:
        data['JD'] = data['JD'] + 2400000.0
    time_obj = Time(data["JD"].to_numpy(), format='jd', scale='tdb')
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


def clean_df(train_df, tol = 3, plot = True, verbose = True):
    '''
    This uses a median absolute deviance filter to remove any crazy outliers in the training data.

    Returns a df with the outliers removed.
    If plot, plots the removed values.
    If verbose, states number of removed values.
    '''
    med = train_df['sind'].median()
    mad = (train_df['sind'] - med).abs().median()
    cleaned_train_df = train_df[(train_df['sind'] - med).abs() < tol * mad]

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
        

    return cleaned_train_df