__author__ = "Enkusellasie Feleke"

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from NCF import NCF
from preprocess import load_data
from config import ROOT
import os

def train_epocs(model,train_df,val_df, epochs=10, lr=0.01, wd=0.0, unsqueeze=False):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    model.train()
    for i in range(epochs):
        users = torch.LongTensor(train_df.userId.values)
        items = torch.LongTensor(train_df.movieId.values)
        ratings = torch.FloatTensor(train_df.rating.values)
        if unsqueeze:
            ratings = ratings.unsqueeze(1)
        y_hat = model(users, items)
        loss = F.mse_loss(y_hat, ratings)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss.item())
    test_loss(model,val_df, unsqueeze)
    return model

def test_loss(model,val_df, unsqueeze=False):
    model.eval()
    users = torch.LongTensor(val_df.userId.values)
    items = torch.LongTensor(val_df.movieId.values)
    ratings = torch.FloatTensor(val_df.rating.values)
    if unsqueeze:
        ratings = ratings.unsqueeze(1)
    y_hat = model(users, items)
    loss = F.mse_loss(y_hat, ratings)
    print("test loss %.3f " % loss.item())

if __name__ == '__main__':
    path = os.path.join(ROOT,'data/ratings.csv')
    train_df, val_df, num_users, num_items = load_data(path)
    model = NCF(num_users, num_items, emb_size=100)
    model = train_epocs(model,train_df,val_df, epochs=15, lr=0.005, wd=1e-6, unsqueeze=True)
    torch.save(model,os.path.join(ROOT,'model.pt'))