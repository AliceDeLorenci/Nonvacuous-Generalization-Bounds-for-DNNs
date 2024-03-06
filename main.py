import numpy as np
import time 
import datetime
import os
from copy import deepcopy
from tqdm import tqdm
import json
import pandas as pd

import torch
from torch import nn, optim
#import torch.nn
#import torch.optim as optim

from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryAccuracy 
from torch.nn.utils import parameters_to_vector

from dataset import BMNIST
from pacbayes import SamplesConvBound, approximate_BPAC_bound, bound_objective, B_RE, quantize_lambda
from loss import Scorer
from models import MLPModel, CNNModel, flip_parameters_to_tensors, set_all_parameters
from parsers import get_main_parser

if __name__ == '__main__':

    args = get_main_parser()
    print(args)
    
    WD = os.getcwd()

    # create timestamped (to avoid overwriting) to save results
    while True:
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") 
        PATH = WD + "/save/{}/".format(timestamp) 
        try:
            os.mkdir(PATH)
            break
        except Exception as e:
            break
    print("Results will be saved in:", PATH)

    # Save arguments for reproducibility
    fname = PATH+"args.json"
    with open(fname, 'w') as file:
        json.dump(args.__dict__, file, indent=4)
    
    # Choose torch device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")     
    print("Using device:", device)


    # Load MNIST dataset
    root = './data/MNIST'
    as_image = True if args.nn_type == 'cnn' else False
    as_binary = True if args.nout == 1 else False
    train_dataset = BMNIST(root+'/train/', train=True, as_image=as_image, as_binary=as_binary, download=True) 
    test_dataset = BMNIST(root+'/test/', train=False, as_image=as_image, as_binary=as_binary, download=True)
        # Split dataset into training and validation sets if use_validation is True
    if args.use_validation:
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    else:
        val_dataset = None

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True, num_workers=args.num_workers) 
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, pin_memory=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, pin_memory=True, num_workers=args.num_workers) if args.use_validation else None

    # Scorer class (defines appropriate loss and accuracy)
    scorer = Scorer(args.nout, device)

    ############################# INITIAL NETWORK TRAINING BY SGD #############################

    # Defining the model (defined inside a method so that it can be re-used to initialize snn)
    def get_model():
        if args.nn_type == 'cnn':
            return CNNModel(args.nin_channels, args.nout, args.nlayers, args.kernel_size, args.nfilters).to(device)
        else:
            return MLPModel(args.nin, args.nlayers, args.nhid, args.nout).to(device)
    model = get_model()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay) 
  
    # SAVE INITIAL SGD PARAMETERS
    fname = PATH+"sgd_model_random_initialization.pt"
    torch.save(model.state_dict(), fname)

    # The random initialization will be used for the prior
    w0 = parameters_to_vector(model.parameters()).to(device)

    print('Starting initial network training by SGD')
    for i in tqdm(range(args.epochs)):
        model.train()
        train_loss = 0
        train_acc = 0 
        for batch in train_loader:
            x, y = batch
            optimizer.zero_grad()

            predictions = model(x.to(device))
            current_loss = scorer.loss(predictions, y.to(device))

            current_loss.backward()
            optimizer.step()
        
            train_loss += current_loss.item()
            train_acc += scorer.accuracy(predictions, y.to(device)).item()
            
        train_acc = train_acc / len(train_loader)
        train_loss = train_loss / len(train_loader)
        
        if args.use_validation:  
            val_loss = 0
            val_acc = 0
            for batch in val_loader:
                x, y = batch

                with torch.no_grad():
                    predictions = model(x.to(device))
                    current_loss = scorer.loss(predictions, y.to(device))
                            
                val_loss += current_loss.item()
                val_acc += scorer.accuracy(predictions, y.to(device)).item()

            val_acc = val_acc / len(val_loader)
            val_loss = val_loss / len(val_loader)
            print('Valid loss', val_loss, 'valid accuracy', val_acc)
        
        model.eval()
        test_loss = 0
        test_acc = 0 
        for batch in test_loader:
            x, y = batch
            
            with torch.no_grad():
                predictions = model(x.to(device))
                current_loss = scorer.loss(predictions, y.to(device))

            test_loss += current_loss.item()
            test_acc += scorer.accuracy(predictions, y.to(device)).item()
            
        test_acc = test_acc / len(test_loader)
        test_loss = test_loss / len(test_loader)
        
        print('Epoch ', str(i+1)
        , ' train loss:' , train_loss 
            , 'test loss', test_loss )
        print('Train accuracy', train_acc, 'test accuracy', test_acc)
        

    # SAVE SGD PARAMETERS
    fname = PATH+"sgd_model.pt"
    torch.save(model.state_dict(), fname)


    # LOAD MODEL
    # fname = PATH+"sgd_model.pt"
    # model.load_state_dict(torch.load(fname))

    ############################# PAC-BAYES BOUND OPTIMIZATION #############################

    # Defining the model
    model_snn = get_model()

    flip_parameters_to_tensors(model_snn)

    # INITIALIZE PARAMETERS TO OPTIMIZE
    # ! Note: parametrisation sigma = 0.5  \log s, \rho = 0.5 \log \lambda
        
    w = parameters_to_vector(model.parameters()).detach().to(device) # PAC-Bayes bound optimization starts from the weights learned by SGD
    rho = torch.Tensor([-3.]).to(device)
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

    if args.scheduler == 'onecycle':
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer_2, max_lr=args.lr2
                                                  , total_steps=args.T, pct_start=args.warmup_pct
                                                  , final_div_factor=1e2)

    # Sample convergence delta (for MC approximation of e(Q,S))
    delta_prime = 0.01

    d = float(len(w))
    m = float(len(train_dataset))

    # Tracking time and initialization for the training loop
    time1 = time.time()
    print_every = 50

    # Interval to save SNN parameters
    save_every=50
    fname = PATH+"snn_model_parameters"

    # Start the training loop
    # model_snn.train()
    loss_ = 0
    count_iter = 0
    best_loss = 1_000_000
    best_params = [w, rho, sigma]
    last_loss_improvement = 0
    
    print('Starting SNN training')
    for t in tqdm(range(args.T)):
        # Update model_snn parameters with the current values of w (with noise added based on sigma)
        noisy_w = w + torch.exp(2 * sigma) * torch.randn_like(w)

        set_all_parameters(model_snn, noisy_w)

        # Compute the PAC-Bayes bound objective
        pb_ = bound_objective(model_snn, train_loader, scorer, w, w0, sigma, rho, d, m, device)

        # Zero gradients, perform backward pass, and update parameters
        optimizer_2.zero_grad()
        pb_.backward()
        optimizer_2.step()
        current_loss = pb_.item()
        loss_ += current_loss
    
        best_loss = min(best_loss, current_loss)
        if args.scheduler != 'none':
            scheduler.step() # NEW 
        
        if best_loss == current_loss:
            best_params = [w, rho, sigma]
            last_loss_improvement = 0
        else:
            last_loss_improvement += 1
            if last_loss_improvement == args.best_loss_patience:
                break
        
        count_iter+= 1
        if count_iter % print_every == 0:
            last_avg_loss = loss_ / print_every
            print(t+1, '/', args.T, 'average loss:' , np.round(last_avg_loss, decimals=4)
                , 'best loss:', np.round(best_loss, decimals=4)
                , '\n ellasped time', time.time() - time1) 
            loss_ = 0

        with open(PATH+"progress.txt", 'a') as file:
            file.write(str(t+1) + '/' + str(args.T) + ' average loss: ' + str(np.round(loss_ / print_every, decimals=4))
                + ' best loss: ' + str(np.round(best_loss, decimals=4))
                + ' ellasped time' + str(time.time() - time1) + '\n')
        
        if count_iter % save_every == 0:
            np.savez_compressed(fname, w=w.detach().cpu().numpy(), sigma=sigma.detach().cpu().numpy(), rho=rho.detach().cpu().numpy()) # SAVE SNN PARAMETERS

    np.savez_compressed(fname, w=w.detach().cpu().numpy(), sigma=sigma.detach().cpu().numpy(), rho=rho.detach().cpu().numpy()) # SAVE SNN PARAMETERS


    #w, rho, sigma = best_params # using best parameters
    
    # Value of rho before quantization of lambda
    rho_old = rho.detach().clone()

    # Quantization of lambda
    rho_plus, rho_minus = quantize_lambda(rho, device)

    empirical_snn_train_errors_ = []
    empirical_snn_test_errors_ = []
    empirical_snn_val_errors_ = []
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
        noisy_w = w + torch.exp(2 * sigma) * torch.randn_like(w)
        set_all_parameters(model_snn, noisy_w)
 
        # compute train accuracy
        train_accuracy = 0
        test_accuracy = 0
        for batch in train_loader:
            x ,y = batch
            with torch.no_grad():
                predictions = model_snn(x.to(device))
                train_accuracy += scorer.accuracy(predictions, y.to(device)).item()*len(x) 
            
        train_accuracy = train_accuracy / len(train_loader.dataset) 
        empirical_snn_train_errors_ += [1- train_accuracy] 

        # compute test accuracy
        for batch in test_loader:
            x, y = batch
            with torch.no_grad(): 
                predictions = model_snn(x.to(device))
                test_accuracy += scorer.accuracy(predictions, y.to(device)).item()*len(x)

        test_accuracy = test_accuracy / len(test_loader.dataset) 
        empirical_snn_test_errors_ += [1 - test_accuracy] 

        if i % print_every == 0:
            print(i , '/ ', args.nb_snns, 'Train error:', np.mean(empirical_snn_train_errors_), 'Test error', np.mean(empirical_snn_test_errors_))
            
    snn_train_error = np.mean(empirical_snn_train_errors_)

    ############################# PAC BAYES BOUND #############################

    bound_1 = SamplesConvBound(snn_train_error, args.nb_snns, delta_prime, ) 
    
    
    B_minus = np.sqrt( 0.5 * B_RE(w, w0, sigma, rho_minus, d, m).item() )
    B_plus = np.sqrt( 0.5 * B_RE(w, w0, sigma, rho_plus, d, m).item() )
    

    pb_bound_prev = bound_1 + min(B_plus, B_minus) 

    # Second bound
    delta = 0.025 # default value, defined in B_RE function
    bound_2 = approximate_BPAC_bound(1-bound_1, min(B_plus, B_minus) )

    number_of_parameters = len(w)                        
    print('Number of parameters:', number_of_parameters)
    print('Train error:', 1-train_acc, 'Test error:', 1-test_acc)
    print('SNN train error:', snn_train_error, 'SNN test error:', np.mean(empirical_snn_test_errors_))
    print('PAC-Bayes bound (before):', pb_bound_prev)
    print('PAC-Bayes bound:', bound_2)
    print('Results saved in', PATH)
    save_dict = {

                'nn_type' : args.nn_type,
                'dataset' : 'MNIST', # enventually change?
            # Architecture
                'nin' : args.nin,
                'nout' : args.nout,
                'nlayers' : args.nlayers,
                 'nhid' : args.nhid,
                 'nb_params' : number_of_parameters,
            # First loop
                 'train_error' : 1-train_acc,
                 'test_error' : 1-test_acc,
                 'val_error' : 1-val_acc,
                 'nn_train_loss' : train_loss,
                 'nn_test_loss' : test_loss,
                 'nn_val_loss' : val_loss,
            # Second loop
                 'snn_train_error' : snn_train_error,
                 'snn_test_error' : np.mean(empirical_snn_test_errors_),
                 'pb_bound_prev' : pb_bound_prev,
                 'pb_bound' : bound_2,
                 'delta_prime' : delta_prime,
                 'delta' : delta,
                  'best_loss_second_loop' : best_loss,
                 'last_avg_loss_second_loop' : last_avg_loss,
                 'nb_snns' : args.nb_snns,
                 'sigma_init' : args.sigma_init,
                 'nb_second_loop_iterations' : args.T, 
            # Hyperparameters
                 'weight_decay' : args.weight_decay,
                 'batch_size' : args.batch_size,
                 'kernel_size' : args.kernel_size,
                 'nin_channels' : args.nin_channels,
                 'nfilters' : args.nfilters,
                 }

    df = pd.DataFrame.from_dict(save_dict, orient='columns')
    df.to_csv(PATH+'results.csv')

    fname = PATH+"results.txt"
    with open(fname, 'w') as file:
        file.write('Number of parameters ' + str(number_of_parameters) + '\n')
        file.write('Train error: ' + str(1-train_acc) + ' Test error: ' + str(1-test_acc) + '\n')
        file.write('SNN train error: ' + str(snn_train_error) + ' SNN test error: ' + str(np.mean(empirical_snn_test_errors_)) + '\n')
        file.write('PAC-Bayes bound (before): ' + str(pb_bound_prev) + '\n')
        file.write('PAC-Bayes bound: ' + str(bound_2) + '\n')
