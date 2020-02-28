'''
This class loads the CIFAR10 data and splits it into training and testing sets
'''

import torch
from torch import nn
import torchvision
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
import matplotlib.pyplot as plt

class Data:

    def __init__(self,gpu,
                      set_name):

        super(Data,self).__init__()

        self.gpu = gpu
        self.set_name = set_name

        # Pull in data

        if self.set_name == "CIFAR10":
            self.transform = transforms.Compose([transforms.ToTensor(), # Images are of size (3,32,32)
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # this will allow us to convert the images into tensors and normalize about 0.5

            self.train_set = torchvision.datasets.CIFAR10(root='./cifardata',
                                                    train=True,
                                                    download=True,
                                                    transform=self.transform)

            self.test_set = torchvision.datasets.CIFAR10(root='./cifardata',
                                                    train=False,
                                                    download=True,
                                                    transform=self.transform)
        elif self.set_name == "MNIST":
            # Pull in data
            self.transform = transforms.Compose([transforms.ToTensor(), # Images are of size (1, 28, 28)
                                            transforms.Normalize((0.1307,), (0.3081,))]) # this will allow us to convert the images into tensors and normalize

            self.train_set = datasets.MNIST(root='../data',
                                            train=True,
                                            download=True,
                                            transform=self.transform)

            self.test_set = torchvision.datasets.MNIST(root='../data',
                                                    train=False,
                                                    download=True,
                                                    transform=self.transform)
        else:
            print("Please enter vaild dataset.")
            exit()
        

        #Test and validation loaders have constant batch sizes, so we can define them 
        #Test and validation loaders have constant batch sizes, so we can define them 
        if self.gpu == False:
            self.test_loader = torch.utils.data.DataLoader(self.test_set,
                                                       batch_size=100,
                                                       shuffle=False)

        else:
            self.test_loader = torch.utils.data.DataLoader(self.test_set,
                                                       batch_size=100,
                                                       shuffle=False,
                                                       num_workers=8,
                                                       pin_memory=True)


        
    # Fucntion to break training set into batches
    def get_train_loader(self,batch_size):
        if self.gpu ==False:
            train_loader = torch.utils.data.DataLoader(self.train_set,
                                                        batch_size = batch_size,
                                                        shuffle = True)
        else:
            train_loader = torch.utils.data.DataLoader(self.train_set,
                                                        batch_size = batch_size,
                                                        shuffle = True,
                                                        num_workers = 8,
                                                        pin_memory=True)

        return(train_loader)

    def unload(self, image_size):
        return image_size.squeeze(0).squeeze(0)

    def get_single_image(self, index = 0,
                               show = False):
        image = self.test_set.data[index]
        image = image[None, None]
        image = image.type('torch.FloatTensor')

        label = self.test_set.targets[index].item()
        label = torch.tensor([label])

        # Display
        if show == True:
            fig, ax = plt.subplots()
            fig.suptitle('Label: ' + str(label.item()), fontsize=16)
            plt.imshow(self.unload(image), cmap='gray')
            plt.show()

        return image, label
