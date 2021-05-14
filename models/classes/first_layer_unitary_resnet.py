'''
This class builds a EfficentNet an optional Unitary Operator on the image
''' 
# Imports
from scipy.stats import ortho_group
from efficientnet_pytorch import EfficientNet
import math
import torch
from torch import nn
import torch.nn.functional as F
from torchsummary import summary
from copy import copy

class FstLayUniResNet(nn.Module):

    def __init__(self, set_name,
                       gpu = False,
                       U = None,
                       model_name = "cifar10_resnet56",
                       pretrained = True):

        super(FstLayUniResNet,self).__init__()

        # Declare Setname 
        self.set_name = set_name

        if self.set_name == "CIFAR10":
            self.image_size = 32
            self.num_classes = 10
        
        else:
            print("Please enter a valid set_name for EffNet.")
            exit()

        # Decalre gpu usage
        self.gpu = gpu

        # Decalre linear encoding
        self.U = U

        # Load EffNet
        self.model_name = model_name

        self.net = torch.hub.load("chenyaofo/pytorch-cifar-models", self.model_name, pretrained=pretrained)

    def display_pretrained_models(self):
        from pprint import pprint
        pprint(torch.hub.list("chenyaofo/pytorch-cifar-models", force_reload=True)) 

    # Generate orthoganal matrix
    def get_orthogonal_matrix(self, size):
        '''
        Generates an orthoganal matrix of input size
        '''
        # Calculate an orthoganal matrix the size of A
        return torch.nn.init.orthogonal_(torch.empty(size, size))

    # Set a new U
    def set_orthogonal_matrix(self):
        self.U = self.get_orthogonal_matrix(self.image_size)

    # Set a new U
    def set_orthogonal_matrix(self):
        self.U = self.get_orthogonal_matrix(self.image_size)
        
    def set_random_matrix(self):
        self.U = torch.rand(self.image_size, self.image_size)

    # Orthogonal transformation
    def orthogonal_operation(self, input_tensor):
        '''
        input tensor A nxn
        generate orthoganal matrix U the size of (n*n)x(n*n)

        Returns UA
        '''
        # Find batch size and feature map size
        batch_size = input_tensor.size(0)
        channel_num = int(input_tensor.size(1))
        A_side_size = int(input_tensor.size(2))

        # Determine if U is available
        if self.U == None:
            return input_tensor
        else:
            U = copy(self.U)

        # Push to GPU if True
        U = copy(U if self.gpu == False else U.cuda())

        # Repeat U and U transpose for all batches
        input_tensor = input_tensor if self.gpu == False else input_tensor.cuda()
        U = copy(U.view((1, A_side_size, A_side_size)).repeat(channel_num * batch_size, 1, 1))
        
        # Batch muiltply UA
        return torch.bmm(U, input_tensor.view(channel_num * batch_size, A_side_size, A_side_size)).view(batch_size, channel_num, A_side_size, A_side_size)

    def forward(self, x):
        
        # Unitary transformation
        x = self.orthogonal_operation(x)

        # Feedforward
        x = self.net(x)

        return x

