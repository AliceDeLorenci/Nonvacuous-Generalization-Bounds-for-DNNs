import numpy as np
import time 
import datetime
import os
from copy import deepcopy
from tqdm import tqdm

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryAccuracy 
from torch.nn.utils import vector_to_parameters, parameters_to_vector

from dataset import BMNIST
from pacbayes import SamplesConvBound, approximate_BPAC_bound, bound_objective, B_RE, quantize_lambda
from loss import logistic, CustomAccuracy
from models import MLPModel
from parsers import get_main_parser

if __name__ == '__main__':

    args = get_main_parser()
    print(args)

    
    PATH = "./save/" # folder to save results
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") # timestamp used on file names to avoid overwriting

    # Choose torch device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")     
    print("Using device:", device)

    ############################# INITIAL NETWORK TRAINING BY SGD #############################

    # Defining the model
    model = MLPModel(args.nin, args.nlayers, args.nhid, args.nout).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9) 
  
    # The random initialization will be used for the prior
    w0 = parameters_to_vector(model.parameters()).to(device)

    # MNIST dataset
    root = './data/MNIST'
    train_dataset = BMNIST(root+'/train/', train=True, download=True) 
    test_dataset = BMNIST(root+'/test/', train=False, download=True)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True, num_workers=args.num_workers) 
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, pin_memory=True, num_workers=args.num_workers)

    print('Starting initial network training by SGD')
    for i in tqdm(range(args.epochs)):
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
            train_acc += CustomAccuracy(predictions, y.to(device)).item()
            
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
            test_acc += CustomAccuracy(predictions, y.to(device)).item()
            
        test_acc = test_acc / len(test_loader)
        
        print('Epoch ', str(i+1)
        , ' train loss:' , train_loss / len(train_loader)
            , 'test loss', test_loss / len(test_loader))
        print('Train accuracy', train_acc, 'test accuracy', test_acc)

    # SAVE SGD PARAMETERS
    fname = PATH+"sgd_model_{}.pt".format(timestamp)
    torch.save(model.state_dict(), fname)

    # LOAD MODEL
    # fname = PATH+"sgd_model.pt"
    # model.load_state_dict(torch.load(fname))

    ############################# PAC-BAYES BOUND OPTIMIZATION #############################

    model_snn = deepcopy(model)

    # INITIALIZE PARAMETERS TO OPTIMIZE
    # ! Note: parametrisation sigma = 0.5  \log s, \rho = 0.5 \log \lambda
        
    w = parameters_to_vector(model.parameters()).detach().to(device) # PAC-Bayes bound optimization starts from the weights learned by SGD
    rho = torch.from_numpy(np.array([-3.]), dtype=np.float32).to(device)
    sigma = 0.5*torch.from_numpy(np.log(1e-6 + args.sigma_init*np.abs(w.detach().cpu().numpy()))).to(device)

    w.requires_grad = True
    rho.requires_grad = True
    sigma.requires_grad = True

    # Convert w to a nn.Parameter if it's not already one
    if not isinstance(w, nn.Parameter):
        w = nn.Parameter(w)

    # Ensure rho and sigma are nn.Parameter objects and on the correct device
    rho = nn.Parameter(rho.to(device))
    sigma = nn.Parameter(sigma.to(device))

    # Initialize the ParameterList and add the parameters
    PB_params = nn.ParameterList([w, rho, sigma])

    # Define the optimizer to update w, rho, and sigma
    optimizer_2 = optim.RMSprop(PB_params, lr=args.lr2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer_2, mode='min', factor=0.2
                                                    ,patience=args.scheduler_patience, min_lr=1e-6)

    # Number of iterations at which to decrease the learning rate to 0.0001
    T_update = 150_000-1 

    # Sample convergence delta (for MC approximation of e(Q,S))
    delta_prime = 0.01  

    d = float(len(w))
    m = float(len(train_dataset))

    # Tracking time and initialization for the training loop
    time1 = time.time()
    print_every = 50

    # Interval to save SNN parameters
    save_every=50
    fname = PATH+"snn_model_parameter_{}".format(timestamp)

    # Start the training loop
    model_snn.train()
    loss_ = 0
    count_iter = 0
    best_loss = 1_000_000
    print('Starting SNN training')
    for t in tqdm(range(args.T)):
        # Update model_snn parameters with the current values of w (with noise added based on sigma)
        noisy_w = w + torch.exp(2 * sigma) * torch.randn_like(w)
        vector_to_parameters(noisy_w, model_snn.parameters())

        # Compute the PAC-Bayes bound objective
        pb_ = bound_objective(model_snn, train_loader, w, w0, sigma, rho, d, m, device)

        # Zero gradients, perform backward pass, and update parameters
        optimizer_2.zero_grad()
        pb_.backward()
        optimizer_2.step()
        loss_ += pb_.item()
    
    best_loss = min(best_loss, pb_.item())
    scheduler.step(best_loss) # NEW 
    
    if t == T_update:
        for g in optimizer_2.param_groups:
            g['lr'] = 1e-4
    
    count_iter+= 1
    if count_iter % print_every == 0:
        print(t+1, '/', T, 'average loss:' , np.round(loss_ / print_every, decimals=2)
              , 'best loss:', np.round(best_loss, decimals=2)
              , '\n ellasped time', time.time() - time1) 
        loss_ = 0
    
    if count_iter % save_every == 0:
        np.savez_compressed(PATH, w=w.detach().cpu().numpy(), sigma=sigma.detach().cpu().numpy(), rho=rho.detach().cpu().numpy()) # SAVE SNN PARAMETERS

    np.savez_compressed(PATH, w=w.detach().cpu().numpy(), sigma=sigma.detach().cpu().numpy(), rho=rho.detach().cpu().numpy()) # SAVE SNN PARAMETERS

    # Value of rho before quantization of lambda
    rho_old = rho.detach().clone()

    # Quantization of lambda
    rho = quantize_lambda(rho, device)

    empirical_snn_train_errors_ = empirical_snn_test_errors_ = []

    #print('Differences between start and end of second loop, w, sigma, rho', torch.norm(w-w_old), torch.norm(sigma-sigma_old))
    #print('Difference before and after discretization of rho', torch.norm(rho-rho_old))

    print('Monte-Carlo Estimation of SNNs accuracies') 
    print_every = 25
    model_snn.eval()

    w.requires_grad = False
    rho.requires_grad = False
    sigma.requires_grad = False

    ############################# MC ESTIMATION OF SNN ERROR #############################

    for i in tqdm(range(args.nb_snns)):

        # sampling the SNN
        vector_to_parameters(w + torch.exp(2*sigma) * torch.randn(w.size()).to(device), model_snn.parameters())
        
        # compute train accuracy
        train_accuracy = 0
        test_accuracy = 0
        for batch in train_loader:
            x ,y = batch
            with torch.no_grad():
                predictions = model_snn(x.to(device))
                train_accuracy += CustomAccuracy(predictions, y.to(device)).item()*len(x) 
            
        train_accuracy = train_accuracy / len(train_loader.dataset) 
        empirical_snn_train_errors_ += [1- train_accuracy] 
        
        # compute test accuracy
        for batch in test_loader:
            x, y = batch
            
            with torch.no_grad(): 
                predictions = model_snn(x.to(device))
                test_accuracy += CustomAccuracy(predictions, y.to(device)).item()*len(x)

        test_accuracy = test_accuracy / len(test_loader.dataset) 
        empirical_snn_test_errors_ += [1 - test_accuracy] 

        if i % print_every == 0:
            print(i , '/ ', args.nb_snns, 'Train error:', np.mean(empirical_snn_train_errors_), 'Test error', np.mean(empirical_snn_test_errors_))
        
    snn_train_error = np.mean(empirical_snn_train_errors_)

    ############################# PAC BAYES BOUND #############################

    bound_1 = SamplesConvBound(snn_train_error, args.nb_snns, delta_prime, ) 

    squared_B = 0.5 * B_RE(w, w0, sigma, rho, d, m).item()
    B = np.sqrt( squared_B )

    pb_bound_prev = bound_1 + B

    bound_2 = approximate_BPAC_bound(1-bound_1-snn_train_error, B)

    number_of_parameters = np.sum([p.numel() for p in model.parameters()])

    print('Number of parameters:', number_of_parameters)
    print('Train error:', 1-train_acc, 'Test error', 1-test_acc)
    print('SNN train error', snn_train_error,  'SNN test error',  np.mean(empirical_snn_test_errors_) )
    print('PAC-Bayes bound (before)', pb_bound_prev )
    print('PAC-Bayes bound', bound_2)


    fname = PATH+"retults_{}.txt".format(timestamp)
    with open(fname, 'w') as file:
        file.write('Number of parameters ' + str(number_of_parameters) + '\n')
        file.write('Train error: ' + str(1-train_acc) + ' Test error ' + str(1-test_acc) + '\n')
        file.write('SNN train error ' + str(snn_train_error) + ' SNN test error ' + str(np.mean(empirical_snn_test_errors_)) + '\n')
        file.write('PAC-Bayes bound (before) ' + str(pb_bound_prev) + '\n')
        file.write('PAC-Bayes bound ' + str(bound_2) + '\n')
