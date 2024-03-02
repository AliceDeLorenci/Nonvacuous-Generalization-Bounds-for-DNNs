import torch
from torch import nn 

# global variables
mu_init = 0
sigma_init = 0.04

def flip_parameters_to_tensors(module):
    attr = []
    while bool(module._parameters):
        attr.append( module._parameters.popitem() )
    setattr(module, 'registered_parameters_name', [])

    for i in attr:
        setattr(module, i[0], torch.zeros(i[1].shape,requires_grad=True))
        module.registered_parameters_name.append(i[0])

    module_name = [k for k,v in module._modules.items()]

    for name in module_name:
        flip_parameters_to_tensors(module._modules[name])

def set_all_parameters(module, theta):
    count = 0  

    for name in module.registered_parameters_name:
        a = count
        b = a + getattr(module, name).numel()
        t = torch.reshape(theta[a:b], getattr(module, name).shape)
        setattr(module, name, t)

        count += getattr(module, name).numel()

    module_name = [k for k,v in module._modules.items()]
    for name in module_name:
        count += set_all_parameters(module._modules[name], theta)
    return count

class CNNModel(nn.Module):
    def __init__(self, nin_channels, nout, 
                 nlayers, kernel_size, nfilters, stride=1, padding=0, padding_mode='zeros', 
                 pool_kernel_size=2, pool_stride=2, 
                 nrow=28, ncol=28):
        super(CNNModel, self).__init__()
        """
        Args:
        - nin_channels: int, number of input channels
        - nout: int, number of output units
        - nlayers: int, number of hidden (convolutional) layers
        - kernel_size: int, size of the convolutional kernel
        - nfilters: int, number of filters
        - stride: int, stride of the convolution
        - padding: int, padding of the convolution
        - padding_mode: string, padding mode of the convolution
        - nrows: int, number of rows of the input image
        - ncols: int, number of columns of the input image
        """

        self.nin_channels = nin_channels
        self.nout = nout

        # Determine output activation function (softmax for multiclass, sigmoid rescaled to [-1,1] for binary)
        if self.nout > 1:
            self.output_activation = lambda x: torch.softmax(x, dim = 1)
        else:
            self.output_activation = lambda x: 2*(torch.sigmoid(torch.squeeze(x))-0.5)

        # Define layers
        self.convlayers = nn.ModuleList()
        self.poollayers = nn.ModuleList()

        self.convlayers.append(nn.Conv2d(nin_channels, nfilters, kernel_size, stride=stride, padding=padding, padding_mode=padding_mode)) # input layer
        self.poollayers.append(nn.MaxPool2d(pool_kernel_size, stride=pool_stride))

        for i in range(nlayers): # hidden layers
            self.convlayers.append(nn.Conv2d(nfilters, nfilters, kernel_size, stride=stride, padding=padding, padding_mode=padding_mode))
            self.poollayers.append(nn.MaxPool2d(pool_kernel_size, stride=pool_stride))
        
        fc_input_channels, fc_input_nrow, fc_input_ncol = self.compute_fc_input_dim(nrow, ncol)

        self.outlayer = nn.Linear(fc_input_channels*fc_input_nrow*fc_input_ncol, nout) # output layer

        # Initialize weights and biases
        for layer in self.convlayers + [self.outlayer]:
            nn.init.trunc_normal_(layer.weight, mean=mu_init, std=sigma_init, a=-2*sigma_init, b=2*sigma_init)
            nn.init.constant_(layer.bias, 0)

        nn.init.constant_(self.convlayers[0].bias, 0.1)

    def forward(self, x):
        for convlayer, poollayer in zip(self.convlayers, self.poollayers):
            x = convlayer(x).relu()
            x = poollayer(x)

        x = x.view(x.size(0), -1)
        x = self.outlayer(x)
        return self.output_activation(x)

    def compute_fc_input_dim(self, nrow=28, ncol=28):
        """
        Computes the input size of the fully connected layer of the encoder block.

        Args:
        - nrow: int, number of rows of the input image
        - ncol: int, number of columns of the input image

        Returns:
        - int, number of channels of the output of the convolutional layers
        - int, number of rows of the output of the convolutional layers
        - int, number of columns of the output of the convolutional layers
        """

        # a meta tensor has no data
        tensor = torch.zeros(1, self.nin_channels, nrow, ncol, device="meta")

        # the tensor is passed through the convolutional layers to determine output size
        for convlayer, poollayer in zip(self.convlayers, self.poollayers):
            tensor = convlayer(tensor)
            tensor = poollayer(tensor)

        return tensor.size(1), tensor.size(2), tensor.size(3)


class MLPModel(nn.Module):
    def __init__(self, nin, nlayers, nhid, nout):

        super(MLPModel, self).__init__()
        
        self.nout = nout
        self.layers = nn.ModuleList()

        # Determine output activation function (softmax for multiclass, sigmoid rescaled to [-1,1] for binary)
        if self.nout > 1:
            self.output_activation = lambda x: torch.softmax(x, dim = 1)
        else:
            self.output_activation = lambda x: 2*(torch.sigmoid(torch.squeeze(x))-0.5)

        # Define layers
        self.layers.append(nn.Linear(nin, nhid)) #input layer

        for i in range(nlayers): #hidden layers
            self.layers.append(nn.Linear(nhid, nhid))

        self.layers.append(nn.Linear(nhid, nout)) #output layer

        # Initialize weights and biases
        for layer in self.layers:
            nn.init.trunc_normal_(layer.weight
                , mean=mu_init, std=sigma_init
                , a=-2*sigma_init, b=2*sigma_init)
            nn.init.constant_(layer.bias, 0)

        nn.init.constant_(self.layers[0].bias, 0.1)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if i+1 == len(self.layers):
                x = layer(x) 
            else:
                x = layer(x).relu()
 
        return self.output_activation(x)
