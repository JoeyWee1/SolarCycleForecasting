import pandas as pd
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