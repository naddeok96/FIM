'''
This class loads the MNIST data and splits it into training and testing sets
'''

import torch
from torch import nn
import torchvision
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms


class MNIST_Data:

    def __init__(self):
        
        # Pull in data
        self.transform = transforms.Compose([transforms.ToTensor(), # Images are of size (1, 28, 28)
                                        transforms.Normalize((0.1307,), (0.3081,))]) # this will allow us to convert the images into tensors and normalize

        self.train_set = torchvision.datasets.MNIST(root='../data',
                                                train=True,
                                                download=True,
                                                transform=self.transform)

        self.test_set = torchvision.datasets.MNIST(root='../data',
                                                train=False,
                                                download=True,
                                                transform=self.transform)

        #Test and validation loaders have constant batch sizes, so we can define them directly
        self.test_loader = torch.utils.data.DataLoader(self.test_set,
                                                       batch_size=100,
                                                       shuffle=False,
                                                       num_workers=8,
                                                       pin_memory=True)

    def get_train_loader(self, batch_size):

        train_loader = torch.utils.data.DataLoader(self.train_set,
                                                    batch_size = batch_size,
                                                    shuffle = True,
                                                    num_workers = 8,
                                                    pin_memory=True)

        return(train_loader)

    def get_single_image(self, index = 0):

        return self.test_loader.dataset.data[index]

