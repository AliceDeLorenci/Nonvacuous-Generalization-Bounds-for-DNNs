import torch
from torch import nn 


class MLPModel(nn.Module):
    def __init__(nin, nlayers, nhid, nout)

        super(MLPModel, self).__init__()

        self.input_layer = nn.Linear(nin, nhid)

        self.hidden_layers = []

        for i in range(nlayers):
            self.hidden_layers += [nn.Linear(nhid, nhid)]

        self.hidden_layers = nn.ModuleList(self.hidden_layers)

        self.output_layer = nn.Linear(nhid, nout)

    def forward(x):

        x = self.input_layer(x).relu()

        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x).relu()

        return self.output_layer(x).relu().softmax(dim=1)

