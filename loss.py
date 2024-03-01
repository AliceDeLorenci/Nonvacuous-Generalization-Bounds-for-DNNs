import torch
import numpy as np
from torchmetrics.classification import BinaryAccuracy, MulticlassAccuracy

class Scorer:
    def __init__(self, nout, device):
        self.nout = nout
        self.device = device

        if self.nout == 1:  # binary classification
            self.loss = self.logistic
            self.accuracy = self.binary_accuracy
        else:               # multiclass classification
            self.loss = torch.nn.CrossEntropyLoss()
            self.accuracy = self.multiclass_accuracy
    
    def logistic(self, predictions, labels):
        return torch.mean( torch.log( 1 + torch.exp(-predictions*labels) ) ) / np.log(2)
    
    def map_to_01(self, x):
        return (x+1)/2

    def binary_accuracy(self, x, y):
        # device = x.device
        acc = BinaryAccuracy(threshold = 0.5).to(self.device)
        return acc( self.map_to_01(x), self.map_to_01(y) )

    def multiclass_accuracy(self, x, y):
        # device = x.device
        acc = MulticlassAccuracy(num_classes=self.nout, average='micro').to(self.device)
        return acc(x, y)


