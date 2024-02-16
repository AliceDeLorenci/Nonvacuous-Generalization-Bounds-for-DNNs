import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from dataset import BMNIST
from torchmetrics.classification import BinaryAccuracy 

import time 
import os
from copy import deepcopy
from tqdm.notebook import tqdm, trange

from torch.nn.utils import vector_to_parameters, parameters_to_vector

from some_functions import get_all_params, Newt, SamplesConvBound, approximate_BPAC_bound
from loss import logistic
from models import MLPModel

from parsers import get_main_parser

args = get_main_parser()

batch_size = 100
nb_epochs = 20



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Accuracy = BinaryAccuracy(threshold = 0.5).to(device)

map_to_01 = lambda x : (x+1) / 2
CumstomAcc = lambda x,y  : Accuracy(map_to_01(x), map_to_01(y))



model = MLPModel(args.nin, args.n_layers, args.nhid, args.nout).to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9) # SGD with the paper's default params

root = './data/MNIST'

w0 = parameters_to_vector(model.parameters())

train_dataset = BMNIST(root+'/train/', train=True, download=True) 
test_dataset = BMNIST(root+'/test/', train=False, download=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=True, num_workers=4)

print('Starting first training loop...')
# first opt loop: classification w logistic loss
for i in trange(nb_epochs):
    model.train()
    train_loss = 0
    train_acc = 0 
    for batch in train_loader:
        x, y = batch
        optimizer.zero_grad()

        predictions = model(x.to(device))
        current_loss = logistic(predictions, y.to(device))

        current_loss.backward()
        optimizer.step()
    
        train_loss += current_loss.item()
        train_acc += CumstomAcc(predictions, y.to(device)).item()
        
    train_acc = train_acc / len(train_loader)
    
    model.eval()
    test_loss = 0
    test_acc = 0 
    for batch in test_loader:
        x, y = batch

        with torch.no_grad():
            predictions = model(x.to(device))
            current_loss = logistic(predictions, y.to(device))

        test_loss += current_loss.item()
        test_acc += CumstomAcc(predictions, y.to(device)).item()
        
    test_acc = test_acc / len(test_loader)
    
    print('Epoch ', str(i+1)
    , ' train loss:' , train_loss / len(train_loader)
        , 'test loss', test_loss / len(test_loader))
    print('Train accuracy', train_acc, 'test accuracy', test_acc)


w = parameters_to_vector(model.parameters()).detach()

# second opt loop optimising the PAC-Bayes bound


nb_snns = 150_000 #nb_snns = 150_000 # number of SNNs to average
T = 200_000 #T = 200_000; 
T_update = 150_000-1 # number of opt iterations
b = 100
c = 0.1
delta = 0.025
delta_prime = 0.01

model_snn = deepcopy(model)

def bound_objective(w, sigma, rho):
    
    return loss(w,sigma) + torch.sqrt(0.5 * B_RE(w,sigma,rho, delta))

def loss(w,sigma, model = model_snn):
    
    vector_to_parameters(w + torch.exp(2*sigma) * torch.randn(w.size()), model.parameters())
   
    loss = torch.from_numpy(np.array([0.0])).to(device)
    
    for batch in train_loader:
        x ,y = batch
        loss += logistic(model(x.to(device)), y.to(device))

    return loss  / len(train_loader)

# ! Note: parametrisastion sigma = 0.5  \log s, \rho = 0.5 \log \lambda
d = len(w); m = len(train_dataset)
def B_RE(w, sigma, rho, delta):
    d = len(w)
    KL = 1/ torch.exp(2*rho)- d + 1 / torch.exp(2*rho) * torch.norm(w-w0) 
    KL = KL / 2  + d * rho -  torch.sum(sigma)
    
    return 1/(m-1) * (KL + 2 * b * np.log(c) - rho*b + np.log( np.pi**2 * m / 6 / delta))


#init parameters to optimise
rho = torch.from_numpy(np.array([-3.]))
sigma = 0.5*torch.from_numpy(np.log(1e-6 + np.abs(w.detach().numpy())))

w.requires_grad = True
rho.requires_grad = True
sigma.requires_grad = True

PB_params = nn.ParameterList()
PB_params.append(w)
PB_params.append(rho)
PB_params.append(sigma)

optimizer_2 = optim.RMSprop(PB_params, lr=1e-3)


for t in trange(T): 
    pb_ = bound_objective(w, sigma, rho)
    
    optimizer_2.zero_grad()
    pb_.backward()
    optimizer_2.step() 
    
    if t == T_update:
        for g in optimizer_2.param_groups:
            g['lr'] = 1e-4
    
lbda = 0.5 * torch.exp(rho).item()
j = int(-b * np.log(lbda / c))
lbda = b * np.exp(- j / c)
rho = np.array( [0.5 * np.log(lbda)])

empirical_snn_train_errors_ = empirical_snn_test_errors_ = []

# sampling SNNs for Monte Carlo estimation 
for i in trange(nb_snns):
    
    vector_to_parameters(w + torch.exp(2*sigma) * torch.randn(w.size()), model_snn.parameters())
    
    train_accuracy = test_accuracy = 0
    for batch in train_loader:
        x ,y = batch
       
        with torch.no_grad(): 
            predictions = model_snn(x.to(device))
            train_accuracy += CumstomAcc(predictions, y.to(device)).item()
        
    empirical_snn_train_errors_ += [1- train_accuracy / len(train_loader)]    
    
    for batch in test_loader:
        x, y = batch
         
        with torch.no_grad(): 
            predictions = model_snn(x.to(device))
            test_accuracy += CumstomAcc(predictions, y.to(device)).item()
        
    empirical_snn_test_errors_ += [1 - test_accuracy / len(test_loader)]

bound_1 = SamplesConvBound(np.mean(empirical_snn_train_errors_), nb_snns)

B = B_RE(w , sigma, torch.from_numpy(rho), delta).item()
bound_2 = approximate_BPAC_bound(bound_1, B)


print('Train error:', 1-train_acc, 'Test error', 1-test_acc)
print('SNN train error', np.mean(empirical_snn_train_errors_),  'SNN test error',  np.mean(empirical_snn_test_errors_) )
print('PAC-Bayes bound', bound_2)
