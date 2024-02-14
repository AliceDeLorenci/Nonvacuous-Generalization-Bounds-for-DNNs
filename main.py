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
w_0 = model.parameters()

train_dataset = BMNIST(root+'/train/', train=True, download=True) 
test_dataset = BMNIST(root+'/test/', train=False, download=True)

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

w = model.parameters()

# second opt loop optimising the PAC-Bayes bound


nb_snns = 150_000 # number of SNNs to average
T = 200_000; T_update = 3 * T // 4 - 1 # number of opt iterations
b = 100
c = 0.1
delta = 0.025


def bound_objective(w, sigma, rho):
    
    return loss(w,sigma) + torch.sqrt(0.5 * B_RE(w,sigma,rho, delta))

def loss(w,sigma, model = model_snn):
    for p, wi, si in zip(model.parameters(), w, sigma):
        p = wi + torch.exp(2*si) * torch.randn(p.size())
   
    loss = torch.Tensor(0.0).to(device)
    
    for batch in train_loader:
        x ,y = batch
        loss += logistic(model(x.to(device)), y.to(device))

    return loss  / len(train_loader)

# ! Note: parametrisastion sigma = 0.5  \log s, \rho = 0.5 \log \lambda

def B_RE(w, sigma, rho, delta):
    
    KL = 1/ torch.exp(2*rho)- d + 1 / torch.exp(2*rho) * torch.norm(w.flatten()-w0) 
    KL = KL / 2  + d * rho -  * sigma
    
    return 1/(m-1) * (KL + 2 * b * torch.log(c) - rho*b + torch.log( torch.pi**2 * m / 6 / delta))
     
    

for i in range(nb_snns):
    model_snn = deepcopy(model)
    
    #init parameters to optimise
    w = model_snn.parameters()
    rho = -3
    sigma = torch.log(2 * np.abs(w))
    
    w.requires_grad = rho.requires_grad = sigma.requires_grad = Trues
    
    optimizer_2 = optim.RMSprop(w, lr=1e-3)
    
    optimizer_2.add_param_group(rho)
    optimizer_2.add_param_group(sigma)
    
    
    for t in range(T):
        
       pb_ = bound_objective(w, sigma, rho)
       
       optimizer_2.zero_grad()
       pb_.backward()
       optimizer_2.step() 
        
        if t == T_update:
            for g in optimizer_2.param_groups:
                g['lr'] = 1e-4
                


    del model_snn