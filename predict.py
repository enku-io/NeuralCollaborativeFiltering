__author__ = "Enkusellasie Feleke"

from config import ROOT
import torch
import os
from NCF import NCF
import numpy as np

def predict(userId):
    path = os.path.join(ROOT, 'model.pt')
    model = torch.load(path)
    model.eval()
    x = np.array(model.getMovies(userId))
    x[::-1].sort()
    return x[0:10]