import torch
from torch import nn 


# global variables
mu_init = 0
sigma_init = 0.04

class MLPModel(nn.Module):
    def __init__(self, nin, nlayers, nhid, nout):

        super(MLPModel, self).__init__()
        
        self.layers = nn.ModuleList()

        self.layers.append(nn.Linear(nin, nhid)) #input layer

        for i in range(nlayers): #hidden layers
            self.layers.append(nn.Linear(nhid, nhid))

        self.layers.append(nn.Linear(nhid, nout)) #output layer

        for layer in self.layers:
            nn.init.trunc_normal_(layer.weight
                , mean=mu_init, std=sigma_init
                , a=-2*sigma_init, b=2*sigma_init)
            nn.init.constant_(layer.bias, 0)

        nn.init.constant_(self.layers[0].bias, 0.1)

    def forward(self, x):

        for layer in self.layers:
            x = layer(x).relu()

        return x.softmax(dim=1)
