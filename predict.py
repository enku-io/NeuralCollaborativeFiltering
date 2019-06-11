__author__ = "Enkusellasie Feleke"

from config import ROOT
import torch
import os
from NCF import NCF
import numpy as np
import operator

def predict(userId):
    path = os.path.join(ROOT, 'model.pt')
    model = torch.load(path)
    model.eval()
    x = model.getMovies(userId)
    return sorted(x.items(), key=operator.itemgetter(1),reverse=True)[0:10]

if __name__ == '__main__':
    print(predict(0))