import torch
from torch import nn, optim

import time 
import os

from parsers import get_main_parser

args = get_main_parser()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)




for i in range(nb_epochs):

    XX
