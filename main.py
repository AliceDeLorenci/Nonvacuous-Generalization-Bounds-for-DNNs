import torch
from torch import nn, optim
from torch.utlis.data import DataLoader, random_split
from dataset import BMNIST

import time 
import os
from copy import deepcopy

from some_functions import get_all_params
from loss import logistic

from parsers import get_main_parser

args = get_main_parser()

batch_size = 100
nb_epochs = 20

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9) # SGD with the paper's default params

root = './data/MNIST'

model = MLPModel(args.n_layers, args.nin, args.nhid, args.nout)
w0 = get_all_params(model)

train_dataset = BMNIST(root+'/train/', train=True, download=True) 
test_dataset = BMNIST(root=root+'/test/', train=False, download=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# first opt loop: classification w logistic loss
for i in range(nb_epochs):
    model.train()
    train_loss = 0
    for batch in train_loader:
        x, y = batch

        predictions = model(x.to(device))
        current_loss = logistic(predictions, y.to(device))

        optimizer.zero_grad(set_to_none=True)
        current_loss.backward()
        optimizer.step()
    
        train_loss += current_loss.item()
    
    model.eval()
    test_loss = 0
    for batch in test_loader:
        x, y = batch

        with torch.no_grad():
            predictions = model(x.to(device))
            current_loss = logistic(predictions, y.to(device))

        test_loss += current_loss.item()

    print('Epoch ', str(i+1)
    , ' train loss:' , train_loss / len(train_loader)
        , 'test loss', test_loss / len(test_loader))
        
    x , y = batch

w = get_all_params(model) 

# second opt loop optimising the PAC-Bayes bound


nb_snns = 150_000 # number of SNNs to average
T = 200_000; T_update = 150_000-1 # number of opt iterations
b = 100
c = 0.1
delta = 0.025

optimizer_2 = optim.RMSprop(, lr=1e-3#TODO)

for i in range(nb_snns):
    model_snn = deepcopy(model)
    # TODO init SNN
    for t in range(T):

        xi = torch.randn(#TODO size )


        if t == T_update:
            learning_rate = 1e-4
            # TODO update optimizer_2 lr
        
