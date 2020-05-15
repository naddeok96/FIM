'''
This class builds a LeNet with specified kernels in each layer and a Random Operator in the last feature map
'''
# Imports
from scipy.stats import ortho_group
import math
import torch
from torch import nn
import torch.nn.functional as F
from torchsummary import summary

class RandLeNet(nn.Module):

    def __init__(self, set_name,
                       gpu = False,
                       num_classes = 10,
                       num_kernels_layer1 = 6, 
                       num_kernels_layer2 = 16, 
                       num_kernels_layer3 = 120,
                       num_nodes_fc_layer = 84):

        super(RandLeNet,self).__init__()

        self.set_name = set_name

        self.gpu = gpu
        
        self.num_classes = num_classes

        self.num_kernels_layer1 = num_kernels_layer1
        self.num_kernels_layer2 = num_kernels_layer2
        self.num_kernels_layer3 = num_kernels_layer3
        self.num_nodes_fc_layer = num_nodes_fc_layer

        if self.set_name == "CIFAR10":
            # Input (3,32,32)
            # Layer 1
            self.conv1 = nn.Conv2d(3, # Input channels
                                self.num_kernels_layer1, # Output Channel 
                                kernel_size = 5, 
                                stride = 1, 
                                padding = 0) # Output = (3,28,28)
        elif self.set_name == "MNIST":
            # Input (1,28,28)
            # Layer 1
            self.conv1 = nn.Conv2d(1, # Input channels
                                self.num_kernels_layer1, # Output Channel 
                                kernel_size = 5, 
                                stride = 1, 
                                padding = 2) # Output = (1,28,28)
        else:
            print("Please enter a valid dataset")
            exit()

        # Layer 2
        self.pool1 = nn.MaxPool2d(kernel_size = 2, 
                                  stride = 2, 
                                  padding = 0) # Output = (num_kernels_layer1,14,14)

        # Layer 3
        self.conv2 = nn.Conv2d(self.num_kernels_layer1,
                               self.num_kernels_layer2,
                               kernel_size = 5, 
                               stride = 1, 
                               padding = 0) # Output = (num_kernels_layer2,10,10)

        # Layer 4
        self.pool2 = nn.MaxPool2d(kernel_size = 2, 
                                  stride = 2, 
                                  padding = 0) # Output = (num_kernels_layer3,5,5)

        # Layer 5
        self.conv3 = nn.Conv2d(self.num_kernels_layer2,
                               self.num_kernels_layer3,
                               kernel_size = 5, 
                               stride = 1, 
                               padding = 0) # Output = (num_kernels_layer4,1,1)

        self.fc1 = nn.Linear(self.num_kernels_layer3, self.num_nodes_fc_layer)

        self.fc2 = nn.Linear(self.num_nodes_fc_layer, self.num_classes)

    # Random transformation
    def random_operation(self, input_tensor):
        '''
        input tensor A 
        generate random matrix R the size of A

        Returns RAR'
        '''
        # Find batch size and feature map size
        num_batches = input_tensor.size()[0]
        A_size = int(math.sqrt(input_tensor.size()[1])) 

        # Calculate an orthoganal matrix the size of A
        R = torch.rand_like(torch.empty(A_size,A_size))

        # Push to GPU if True
        R = R if self.gpu == False else R.cuda()

        # Repeat R and R transpose for all batches
        input_tensor = input_tensor if self.gpu == False else input_tensor.cuda()
        Rt = R.t().view((1, A_size, A_size)).repeat(num_batches, 1, 1)
        R = R.view((1, A_size, A_size)).repeat(num_batches, 1, 1)
        
        # # Batch muiltply UAU'
        return torch.bmm(torch.bmm(R, # U
                                   input_tensor.view(num_batches, A_size, A_size)), # A resized
                                   Rt).view(num_batches, A_size**2, 1 ,1) # Ut then resize output


    def forward(self, x):
        
        x = torch.tanh(self.conv1(x))
        x = self.pool1(x)
        x = torch.tanh(self.conv2(x))
        x = self.pool2(x)
        x = torch.tanh(self.conv3(x))
        x = self.random_operation(x)
        x = x.view(-1, self.num_kernels_layer3 * 1 * 1)
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)

        return x

