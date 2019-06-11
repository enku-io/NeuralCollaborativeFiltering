__author__ = "Enkusellasie Feleke"

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def proc_col(col, train_col=None):
    """Encodes a pandas column with continous ids.
    """
    if train_col is not None:
        uniq = train_col.unique()
    else:
        uniq = col.unique()
    name2idx = {o: i for i, o in enumerate(uniq)}
    return name2idx, np.array([name2idx.get(x, -1) for x in col]), len(uniq)


def encode_data(df, train=None):
    """ Encodes rating data with continous user and movie ids.
    If train is provided, encodes df with the same encoding as train.
    """
    df = df.copy()
    for col_name in ["userId", "movieId"]:
        train_col = None
        if train is not None:
            train_col = train[col_name]
        _, col, _ = proc_col(df[col_name], train_col)
        df[col_name] = col
        df = df[df[col_name] >= 0]
    return df

def split_data(d,percent=0.8):
    np.random.seed(3)
    msk = np.random.rand(len(d)) < percent
    train = d[msk].copy()
    val = d[~msk].copy()
    return train, val

def load_data(path):
    data = pd.read_csv(path)
    train, val = split_data(data)
    train_df = encode_data(train)
    val_df = encode_data(val, train)
    num_users = len(train.userId.unique())
    num_items = len(train.movieId.unique())
    return train_df,val_df,num_users,num_items