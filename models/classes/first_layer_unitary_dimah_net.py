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

        super(FstLayUniLeNet,self).__init__()

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

        # Layer Dictionary
        # Input 3x32x32
        self.layer = {"Conv" : {  1 : { "in"  : 3,
                                    "out" : 32,
                                    "kernel_size" : 5,
                                    "stride" : 1,
                                    "padding" : 0},
                                    # Output 32x28x28

                                    ## Batch Norm 1 ##

                                    ## MaxPooling1 ##
                                    # Output 32x27x27

                                2 : { "in"  : 32,
                                    "out" : 32,
                                    "kernel_size" : 3,
                                    "stride" : 1,
                                    "padding" : 1},
                                    # Output 32x27x27

                                    ## Batch Norm 2 ##

                                3 : { "in"  : 32,
                                    "out" : 32,
                                    "kernel_size" : 3,
                                    "stride" : 1,
                                    "padding" : 1},
                                    # Output 32x27x27

                                    ## Batch Norm 3 ##

                                    ## MaxPooling2 ##
                                    # Output 32x26x26

                                4 : { "in"  : 32,
                                    "out" : 32,
                                    "kernel_size" : 3,
                                    "stride" : 1,
                                    "padding" : 1},
                                    # Output 32x26x26

                                    ## Batch Norm 4 ##

                                5 : { "in"  : 32,
                                    "out" : 64,
                                    "kernel_size" : 3,
                                    "stride" : 1,
                                    "padding" : 1},
                                    # Output 64x26x26

                                    ## Batch Norm 5 ##

                                    ## MaxPooling3 ##
                                    # Output 64x25x25

                                6 : { "in"  : 64,
                                    "out" : 64,
                                    "kernel_size" : 3,
                                    "stride" : 1,
                                    "padding" : 1},
                                    # Output 64x25x25

                                    ## Batch Norm 6 ##

                                7 : { "in"  : 64,
                                    "out" : 64,
                                    "kernel_size" : 3,
                                    "stride" : 1,
                                    "padding" : 1},
                                    # Output 64x25x25

                                    ## Batch Norm 7 ##

                                    ## MaxPooling4 ##
                                    # Output 64x24x24

                                8 : { "in"  : 64,
                                    "out" : 128,
                                    "kernel_size" : 3,
                                    "stride" : 1,
                                    "padding" : 1},
                                    # Output 128x24x24

                                    ## Batch Norm 8 ##

                                9 : { "in"  : 128,
                                    "out" : 128,
                                    "kernel_size" : 3,
                                    "stride" : 1,
                                    "padding" : 1},
                                    # Output 128x24x24

                                    ## Batch Norm 9 ##

                                    ## MaxPooling5 ##
                                    # Output 128x23x23

                                10 : { "in"  : 128,
                                    "out"   : 128,
                                    "kernel_size" : 1,
                                    "stride" : 1,
                                    "padding" : 0}},
                                    # Output 128x23x23

               "Pool" : {1 :    {"kernel_size" : 2,
                                "stride" : 1,
                                "padding" : 0},
                         2 :    {"kernel_size" : 2,
                                "stride" : 1,
                                "padding" : 0},
                         3 :    {"kernel_size" : 2,
                                "stride" : 1,
                                "padding" : 0},
                         4 :    {"kernel_size" : 2,
                                "stride" : 1,
                                "padding" : 0},
                         5 :    {"kernel_size" : 2,
                                "stride" : 1,
                                "padding" : 0}},

               "BatchNorm" : {1 : {"in"   :  32},
                              2 : {"in"   :  32},
                              3 : {"in"   :  32},
                              4 : {"in"   :  32},
                              5 : {"in"   :  64},
                              6 : {"in"   :  64},
                              7 : {"in"   :  64},
                              8 : {"in"   :  128},
                              9 : {"in"   :  128}},

               "FC"        : {1 : {"in"  : 23*23*128,
                                   "out" : 10}}
               
        }
        
        self.load_layer_functions

        
    def load_layer_functions(self):
            self.conv      = {}
            self.pool      = {}
            self.batchnorm = {}
            self.fc        = {}
            for layer_type in self.layer:
                for i in self.layer[layer_type]:
                    layer_data = self.layer[layer_type][i]

                    if layer_type == "Conv":
                        self.conv.update{i + 1: nn.Conv2d( layer_data["in"], layer_data["out"],
                                                        kernel_size = layer_data["kernel_size"], 
                                                        stride = layer_data["stride"], 
                                                        padding = layer_data["padding"])} 

                    elif layer_type == "Pool":
                        self.pool.update{i + 1: nn.MaxPool2d(kernel_size = layer_data["kernel_size"], 
                                                                     stride = layer_data["stride"], 
                                                                     padding = layer_data["padding"])}

                    elif layer_type == "BatchNorm":
                        self.batchnorm.update{i + 1: nn.BatchNorm2d(layer_data["in"])}

                    elif layer_type == "FC":
                        self.fc.update{i + 1 : nn.Linear(layer_data["in"], 
                                                         layer_data["out"])}

                    else:
                        print("Invalid Layer Type")
                        exit()

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
        x = self.elu(self.conv[1](x))
        x = self.batch_norm[1](x)
        x = self.pool[1](x)

        # 4 Intermediate Blocks
        for i in range(4):
            x = self.elu(self.conv[2 + 2*i](x))
            x = self.batch_norm[2 + 2*i](x)
            x = self.elu(self.conv[3 + 2*i](x))
            x = self.batch_norm[3 + 2*i](x)
            x = self.pool[2 + i](x)
            x = self.dropout(x)

        # Outro Block
        x = self.elu(self.conv[10](x))
        x = self.batch_norm[9](x)
        x = self.pool[6](x)

        # Flatten and Feed to FC
        x = x.view(-1, 1)
        x = self.fc[1](x)
        x = self.soft_max(x)

        return x

