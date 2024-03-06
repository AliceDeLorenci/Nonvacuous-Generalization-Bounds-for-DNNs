import numpy as np

import torch
from torch import nn, optim
from torch.utils.data import DataLoader



from dataset import BMNIST
from loss import Scorer
from models import MLPModel, CNNModel
from utils import vec2params

from parsers import get_lottery_parser

if __name__ == "__main__":
   
   # params to define
   
    # nb of pruning rounds
    # mask 
    # nb of epochs
    # learning rate
    # batch size
    # number of workers

    args = get_lottery_parser()
    
        # create timestamped (to avoid overwriting) to save results
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") 
    PATH = "./save/{}/".format(timestamp)                         
    # PATH = "./save/only_loss_term/"                                ## !!!
    os.makedirs(PATH, exist_ok=True)

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

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True, num_workers=args.num_workers) 
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, pin_memory=True, num_workers=args.num_workers)

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

    for round in range(args.pruning_rounds):
        if round > 0:
            model = get_model()

            # code MP step 
            
            # apply mask to the model

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
            
            print('Epoch ', str(i+1)
            , ' train loss:' , train_loss / len(train_loader)
                , 'test loss', test_loss / len(test_loader))
            print('Train accuracy', train_acc, 'test accuracy', test_acc)

    # SAVE SGD PARAMETERS
    # save model 
    
    # LOAD MODEL
    # fname = PATH+"sgd_model.pt"
    # model.load_state_dict(torch.load(fname))
    