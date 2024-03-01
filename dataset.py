from torchvision.datasets import MNIST
from torch.utils.data import Dataset
import torchvision.transforms as transforms 
import torch


class BMNIST(MNIST):
    '''
    Class for MNIST dataset with binarised labels
    '''
    def __init__(self, root, train, as_image=False, as_binary=True, download=True):

        if as_image:
            transform = transforms.Compose([
                transforms.PILToTensor(),
                transforms.ConvertImageDtype(torch.float32)
                ])
        else:
            transform = transforms.Compose([
                transforms.PILToTensor(),
                transforms.ConvertImageDtype(torch.float32),
                transforms.Lambda(lambda x: torch.flatten(x)),
                ])

        if as_binary:
            target_transform = lambda x: 2*int(x < 5)-1
        else:
            target_transform = None

        super(BMNIST, self).__init__(root, train, transform, target_transform, download)