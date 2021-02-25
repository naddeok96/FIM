'''
This class builds a LeNet with specified kernels in each layer and Unitary Operator on the image
'''
# Imports
from scipy.stats import ortho_group
import torchvision.models as models
import math
import torch
from torch import nn
import torch.nn.functional as F
from torchsummary import summary
from copy import copy

class FstLayUniVGG16(nn.Module):

    def __init__(self, gpu = False,
                       U = None,
                       seed = 1):

        super(FstLayUniVGG16,self).__init__()
        torch.manual_seed(seed)

        # Decalre Hyperparameters
        self.gpu = gpu

        self.U = U

        self.vgg = models.vgg16(pretrained=True if U is None else False)


    # Generate orthoganal matrix
    def get_orthogonal_matrix(self, size):
        '''
        Generates an orthoganal matrix of input size
        '''
        # Calculate an orthoganal matrix the size of A
        return torch.nn.init.orthogonal_(torch.empty(size, size))

    # Set a new U
    def set_orthogonal_matrix(self):
        self.U = self.get_orthogonal_matrix(self.image_size**2)

    # Orthogonal transformation
    def orthogonal_operation(self, input_tensor):
        '''
        input tensor A nxn
        generate orthoganal matrix U the size of (n*n)x(n*n)

        Returns UA
        '''
        # Find batch size and feature map size
        print(input_tensor.size())
        exit()
        batch_size = input_tensor.size(0)
        channel_num = int(input_tensor.size(1))
        A_side_size = int(input_tensor.size(2))

        # Determine if U is available
        U = copy(self.U)

        # Push to GPU if True
        U = copy(U if self.gpu == False else U.cuda())

        # Repeat U and U transpose for all batches
        input_tensor = input_tensor if self.gpu == False else input_tensor.cuda()
        # Ut = U.t().view((1, A_side_size**2, A_side_size**2)).repeat(batch_size, 1, 1)
        U = copy(U.view((1, A_side_size**2, A_side_size**2)).repeat(channel_num * batch_size, 1, 1))
        
        # Batch muiltply UA
        return torch.bmm(U, input_tensor.view(channel_num *batch_size, A_side_size**2, 1)).view(batch_size, channel_num, A_side_size, A_side_size)

    def forward(self, x):
        
        # Unitary transformation
        if self.U is not None:
            x = self.orthogonal_operation(x)

        # Feedforward
        x = self.vgg(x)

        return x

