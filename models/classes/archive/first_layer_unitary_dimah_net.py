'''
This class builds a DimahNet with specified kernels in each layer and Unitary Operator on the image
'''
# Imports
from scipy.stats import ortho_group
import math
import torch
from torch import nn
import torch.nn.functional as F
from torchsummary import summary
from copy import copy

class FstLayUniDimahNet(nn.Module):

    def __init__(self, gpu,
                       U = None):

        super(FstLayUniDimahNet,self).__init__()

        # Decalre GPU usage
        self.gpu = gpu

        # Load U
        self.U = U

        # Actvation Function
        self.elu = torch.nn.ELU()

        # Dropout Function
        self.dropout = torch.nn.Dropout(p = 0.2)

        # Softmax Function
        self.soft_max = torch.nn.Softmax(dim = 1)

        # Image Size
        self.image_size = 32


        # Layer Dictionary
        # Intro Block
        self.conv1 = nn.Conv2d( 3, 32,
                                kernel_size = 5, 
                                stride = 1, 
                                padding = 0)
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size = 2, 
                                    stride = 1, 
                                    padding = 0)

        # Intermediate Block1
        self.conv2 = nn.Conv2d( 32, 32,
                                kernel_size = 3, 
                                stride = 1, 
                                padding = 1)
        self.batch_norm2 =  nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d( 32, 32,
                                kernel_size = 3, 
                                stride = 1, 
                                padding = 1)
        self.batch_norm3 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size = 2, 
                                    stride = 1, 
                                    padding = 0)
            
        # Intermediate Block2
        self.conv4 = nn.Conv2d( 32, 32,
                                kernel_size = 3, 
                                stride = 1, 
                                padding = 1)
        self.batch_norm4 = nn.BatchNorm2d(32)
        self.conv5 = nn.Conv2d( 32, 64,
                                kernel_size = 3, 
                                stride = 1, 
                                padding = 1)
        self.batch_norm5 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size = 2, 
                                stride = 1, 
                                padding = 0)

        # Intermediate Block3
        self.conv6 = nn.Conv2d( 64, 64,
                                kernel_size = 3, 
                                stride = 1, 
                                padding = 1)
        self.batch_norm6 = nn.BatchNorm2d(64)
        self.conv7 = nn.Conv2d( 64, 64,
                                kernel_size = 3, 
                                stride = 1, 
                                padding = 1)
        self.batch_norm7 = nn.BatchNorm2d(64)
        self.pool4 = nn.MaxPool2d(kernel_size = 2, 
                                    stride = 1, 
                                    padding = 0)

        # Intermediate Block4
        self.conv8 = nn.Conv2d( 64, 128,
                                kernel_size = 3, 
                                stride = 1, 
                                padding = 1)
        self.batch_norm8 = nn.BatchNorm2d(128)
        self.conv9 = nn.Conv2d( 128, 128,
                                kernel_size = 3, 
                                stride = 1, 
                                padding = 1)
        self.batch_norm9 = nn.BatchNorm2d(128)
        self.pool5 = nn.MaxPool2d(kernel_size = 2, 
                                    stride = 1, 
                                    padding = 0)

        # Outro Block
        self.conv10 = nn.Conv2d( 128, 128,
                                kernel_size = 1, 
                                stride = 1, 
                                padding = 0)
        self.batch_norm10 = nn.BatchNorm2d(128)
        self.pool6 = nn.MaxPool2d(kernel_size = 2, 
                                    stride = 1, 
                                    padding = 0)

        # FC
        self.fc1 = nn.Linear(22*22*128, 10)

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

        # Intro Block
        x = self.elu(self.conv1(x))
        x = self.batch_norm1(x)
        x = self.pool1(x)

        # Intermediate Block1
        x = self.elu(self.conv2(x))
        x = self.batch_norm2(x)
        x = self.elu(self.conv3(x))
        x = self.batch_norm3(x)
        x = self.pool2(x)
        x = self.dropout(x)
            
        # Intermediate Block2
        x = self.elu(self.conv4(x))
        x = self.batch_norm4(x)
        x = self.elu(self.conv5(x))
        x = self.batch_norm5(x)
        x = self.pool3(x)
        x = self.dropout(x)

        # Intermediate Block3
        x = self.elu(self.conv6(x))
        x = self.batch_norm6(x)
        x = self.elu(self.conv7(x))
        x = self.batch_norm7(x)
        x = self.pool4(x)
        x = self.dropout(x)

        # Intermediate Block4
        x = self.elu(self.conv8(x))
        x = self.batch_norm8(x)
        x = self.elu(self.conv9(x))
        x = self.batch_norm9(x)
        x = self.pool5(x)
        x = self.dropout(x)

        # Outro Block
        x = self.elu(self.conv10(x))
        x = self.batch_norm10(x)
        x = self.pool6(x)

        # Flatten and Feed to FC
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.soft_max(x)

        return x

