'''
This class builds a NN an optional Unitary Operator on the image
''' 
# Imports
import torch
from torch import nn
from copy import copy
import sys
sys.path.append(".")
from .adjustable_lenet import AdjLeNet

class FstLayUniNet(nn.Module):

    def __init__(self, set_name,
                       gpu        = False,
                       U_filename = None,
                       model_name = None,
                       pretrained = None):

        super(FstLayUniNet,self).__init__()

        # Declare Setname 
        self.set_name = set_name

        if self.set_name == "CIFAR10":
            self.num_channels = 3
            self.image_size   = 32
            self.num_classes  = 10

        elif self.set_name == "MNIST":
            self.num_channels = 1
            self.image_size   = 28
            self.num_classes  = 10

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
        if self.set_name == "MNIST":
            if model_name == "lenet":
                self.model_name = model_name
                
                self.net =  AdjLeNet(set_name   = self.set_name, 
                                    num_classes = self.num_classes,
                                    pretrained_weights_filename= "models/pretrained/MNIST/LeNet_Attacker_w_acc_98.pt" if pretrained else None)

                self.accuracy = 98 if pretrained else None
            else:
                print("Only lenet is available for MNIST")
                exit()

        elif self.set_name == "CIFAR10":
            self.model_name = model_name

            self.net = torch.hub.load("chenyaofo/pytorch-cifar-models", self.model_name, pretrained=pretrained, verbose = False)

        else:
            print("Please enter vaild dataset. (Options: MNIST, CIFAR10)")
            exit()

        # Organize GPUs
        if isinstance(self.gpu, bool):
            # Push to GPU if True
            self.net = self.net.cuda() if self.gpu else self.net

        else:
            # Push to rank of gpu
            self.net = self.net.to(gpu)
        
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

        # Organize GPUs
        if isinstance(self.gpu, bool):
            # Push to GPU if True
            self.U = self.U.cuda() if self.gpu else self.U

        else:
            # Push to rank of gpu
            self.U = self.U.to(gpu)
        
    def load_U_from_Ufilename(self):
        # Decode filename for stats
        stats = [""] * (2 * self.num_channels) # Initalize a spot for mean and std on each channel
        state = 0                              # Declare initial state
        middle_state = self.num_channels + 1   # Declare state between means and stds
        sofar = ""                             # Declare charcters seen so far in title

        # Iterate through each character in title
        for c in self.U_filename:   
            # Add Character
            sofar += c

            # If the intro is finished begin on first channel mean
            if sofar == "models/pretrained/" + self.set_name + "/U_w_means_":
                state = 1
                continue
            
            # If the middle state is reached do nothing untill "and_stds_" is read
            if state == middle_state and "and_stds_" not in sofar:
                continue

            # If the current state is middle ignore the first "_" without increasing the state
            if state == middle_state and len(stats[state - 1]) == 0 and c == "_":
                continue

            # Decode
            if 0 < state and state < len(stats) + 1:
                # "n" represents a minus symbol
                if c == "n":
                    stats[state - 1] += "-"

                # "-" represents a "." symbol
                elif c == "-":
                    stats[state - 1] += "."

                # "_" represents the end of the current stat
                elif c == "_":
                    stats[state - 1] = float(stats[state-1])
                    state += 1

                # Otherwise the character should be stored with no decoding
                else: 
                    stats[state - 1] += c

        # Seperate Stats
        self.U_means = torch.tensor(stats[0:self.num_channels])
        self.U_stds  = torch.tensor(stats[self.num_channels:])

        # Load U 
        self.U = torch.load(self.U_filename)

        # Push to GPU
        # Organize GPUs
        if isinstance(self.gpu, bool):
            # Push to GPU if True
            if self.gpu:
                self.U       = self.U.cuda()
                self.U_means = self.U_means.cuda()
                self.U_stds  = self.U_stds.cuda()

        else:
            # Push to rank of gpu
            self.U       = self.U.to(self.gpu)
            self.U_means = self.U_means.to(self.gpu)
            self.U_stds  = self.U_stds.to(self.gpu)

        print(self.U_filename, " is Loaded.")

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
        batch_size  = input_tensor.size(0)
        channel_num = int(input_tensor.size(1))
        A_side_size = int(input_tensor.size(2))

        # Determine if U is available
        if self.U is None:
            return input_tensor

        else:
            U = copy(self.U)

        # If gpu is not bool the DDP is being used
        if isinstance(self.gpu, bool):
            # Push to GPU if True
            U = U.cuda() if self.gpu else U
            input_tensor = input_tensor.cuda() if self.gpu else input_tensor

        else:
            # Push to rank of gpu
            U = U.to(self.gpu)
            input_tensor = input_tensor.to(self.gpu)
            
        # Repeat U and U transpose for all batches
        U = U.view((1, A_side_size, A_side_size)).repeat(channel_num * batch_size, 1, 1)
        
        # Batch muiltply UA
        UA = torch.bmm( U, 
                        input_tensor.view(channel_num * batch_size, A_side_size, A_side_size)
                        ).view(batch_size, channel_num, A_side_size, A_side_size)

        
        # Normalize
        if self.normalize_U:
            UA = UA.view(UA.size(0), UA.size(1), -1)
            batch_means = self.U_means.repeat(UA.size(0), 1).view(UA.size(0), UA.size(1), 1)
            batch_stds  = self.U_stds.repeat(UA.size(0), 1).view(UA.size(0), UA.size(1), 1)

            return UA.sub_(batch_means).div_(batch_stds).view(UA.size(0), UA.size(1), self.image_size, self.image_size)

        else:
            return UA

    def forward(self, x):
        
        # Unitary transformation
        x = self.orthogonal_operation(x)

        # Feedforward
        x = self.net(x)

        return x

