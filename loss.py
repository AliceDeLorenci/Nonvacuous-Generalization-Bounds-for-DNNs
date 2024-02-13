import torch

log_el = lambda y, yhat : torch.log(1 + torch.exp(- y*yhat)) / torch.log(2)

def logistic(predictions, labels):

    return torch.mean(log_el(predictions, labels))