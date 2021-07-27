'''
This class builds a NN an optional Unitary Operator on the image
''' 
# Imports
import torch
from torch import nn
from copy import copy

class FstLayUniNet(nn.Module):

    def __init__(self, set_name,
                       gpu = False,
                       U_filename = None,
                       model_name = "cifar10_resnet56",
                       pretrained = True):

        super(FstLayUniNet,self).__init__()

        # Declare Setname 
        self.set_name = set_name

        if self.set_name == "CIFAR10":
            self.image_size = 32
            self.num_classes = 10

        elif self.set_name == "MNIST":
            self.image_size = 28
            self.num_classes = 10

        else:
            print("Please enter a valid set_name for EffNet.")
            exit()

        # Decalre gpu usage
        self.gpu = gpu

        # Declare U
        self.U_filename = U_filename
        if U_filename is None:
            self.U = None
            self.U_means = None
            self.U_stds  = None
            self.normalize_U = False

        else:
            self.load_U_from_Ufilename()
            self.normalize_U = True
            
        # Load Net
        self.model_name = model_name

        self.net = torch.hub.load("chenyaofo/pytorch-cifar-models", self.model_name, pretrained=pretrained, verbose = False)

        self.net = self.net.cuda() if self.gpu else self.net

    # Generate orthoganal matrix with no Mean or STD Stats
    def get_orthogonal_matrix(self, size):
        '''
        Generates an orthoganal matrix of input size
        '''
        # Calculate an orthoganal matrix the size of A
        return torch.nn.init.orthogonal_(torch.empty(size, size))

    # Set a new U with no Mean or STD Stats
    def set_orthogonal_matrix(self):
        self.U = self.get_orthogonal_matrix(self.image_size)

        self.U = self.U.cuda() if self.gpu else self.U

    def load_U_from_Ufilename(self):
        # Decode filename for stats
        stats = ["", "", "", "", "", ""]
        state = 0
        sofar = ""
        for c in self.U_filename:        
            sofar += c
            if sofar == "models/pretrained/U_w_means_":
                state = 1
                continue
            
            if state == 4 and "and_stds_" not in sofar:
                continue

            if state == 4 and len(stats[state - 1]) == 0 and c == "_":
                continue

            if 0 < state and state < 7:
                if c == "n":
                    stats[state - 1] += "-"
                elif c == "-":
                    stats[state - 1] += "."
                elif c == "_":
                    stats[state - 1] = float(stats[state-1])
                    state += 1
                else: 
                    stats[state - 1] += c

        # Seperate Stats
        self.U_means = torch.tensor(stats[0:3])
        self.U_stds  = torch.tensor(stats[3:])

        # Load U 
        self.U = torch.load(self.U_filename)

        # Push to GPU
        if self.gpu:
            self.U       = self.U.cuda()
            self.U_means = self.U_means.cuda()
            self.U_stds  = self.U_stds.cuda()

    def display_pretrained_models(self):
        from pprint import pprint
        pprint(torch.hub.list("chenyaofo/pytorch-cifar-models", force_reload=True)) 

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
        U = copy(U.cuda() if self.gpu else U)

        # Repeat U and U transpose for all batches
        input_tensor = input_tensor.cuda() if self.gpu else input_tensor

        U = copy(U.view((1, A_side_size, A_side_size)).repeat(channel_num * batch_size, 1, 1))
        
        # Batch muiltply UA
        UA = torch.bmm(U, 
                       input_tensor.view(channel_num * batch_size, A_side_size, A_side_size)
        ).view(batch_size, channel_num, A_side_size, A_side_size)

        
        # Normalize
        if self.normalize_U:
            UA = UA.view(UA.size(0), UA.size(1), -1)
            batch_means = self.U_means.repeat(UA.size(0), 1).view(UA.size(0), UA.size(1), 1)
            batch_stds  = self.U_stds.repeat(UA.size(0), 1).view(UA.size(0), UA.size(1), 1)
            return UA.sub_(batch_means).div_(batch_stds).view(UA.size(0), UA.size(1), 32, 32)

        else:
            return UA

    def forward(self, x):
        
        # Unitary transformation
        x = self.orthogonal_operation(x)

        # Feedforward
        x = self.net(x)

        return x

