import torch
from torch import nn, optim
from torch.utlis.data import DataLoader
from dataset import BMNIST

import time 
import os

from loss import logistic

from parsers import get_main_parser

args = get_main_parser()

batch_size = 100
nb_epochs = 20

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)


dataset = BMNIST() # TODO args
train_loader = DataLoader(dataset, batch_size=batch_size)


# first opt loop: classification w logistic loss
for i in range(nb_epochs):

    for batch in train_loader:

        x, y = batch
    


w = # TODO get model weights

# second opt loop optimising the PAC-Bayes bound



nb_snns = 15_000
T = 200_000; T_update = 150_000-1
b = # TODO
c = # TODO

optimizer_2 = optim.RMSprop(, lr=1e-3#TODO)

for i in range(nb_snns):
    for t in range(T):
        
        xi = torch.randn(#TODO size )


        if t == T_update:
            learning_rate = 1e-4
            # TODO update optimizer_2 lr
        
