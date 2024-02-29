import torch
import numpy as np
from torchmetrics.classification import BinaryAccuracy 

# element-wise logistic loss
log_el = lambda yhat, y : torch.log(1 + torch.exp(- y*yhat)) / np.log(2)

def logistic(predictions, labels):
    return torch.mean( torch.log( 1 + torch.exp(-predictions*labels) ) ) / np.log(2)

def map_to_01(x):
    return (x+1)/2

def CustomAccuracy(x, y):
    device = x.device
    Accuracy = BinaryAccuracy(threshold = 0.5).to(device)
    return Accuracy(map_to_01(x), map_to_01(y))