'''
This class builds a LeNet with specified kernels in each layer and Unitary Operator on the image
'''
# Imports
from scipy.stats import ortho_group
import math
import torch
from torch import nn
import torch.nn.functional as F
from torchsummary import summary
from copy import copy

class FstLayUniDenseNet(nn.Module):

    def __init__(self, set_name,
                       gpu = False,
                       U = None,
                       num_nodes_fc_layer2 = 1024,
                       num_nodes_fc_layer3 = 512,
                       num_classes = 10,
                       pretrained_weights_filename = None,
                       pretrained_unitary_matrix_filename = None,
                       seed = 1):

        super(FstLayUniDenseNet,self).__init__()
        torch.manual_seed(seed)

        # Decalre Hyperparameters
        self.set_name = set_name

        self.gpu = gpu

        self.U = U

        self.num_nodes_fc_layer2 = num_nodes_fc_layer2
        self.num_nodes_fc_layer3 = num_nodes_fc_layer3
        self.num_classes = num_classes

        if self.set_name == "CIFAR10":
            # Input (3,32,32)
            self.flat_size = 3 * 32 * 32
      
        elif self.set_name == "MNIST":
            # Input (1,28,28)
            self.flat_size = 28 * 28

        else:
            print("Please enter a valid dataset")
            exit()

        # Layer 1
        self.fc1 = nn.Linear(self.flat_size, self.num_nodes_fc_layer2)

        # Layer 2
        self.fc2 = nn.Linear(self.num_nodes_fc_layer2, self.num_nodes_fc_layer3)

        # Layer 3
        self.fc3 = nn.Linear(self.num_nodes_fc_layer3, self.num_classes)

        # Load pretrained parameters
        if pretrained_unitary_matrix_filename is not None:
            self.U = torch.load(pretrained_unitary_matrix_filename, map_location=torch.device('cpu'))
        if pretrained_weights_filename is not None:
            self.load_state_dict(torch.load(pretrained_weights_filename, map_location=torch.device('cpu')))
        if (pretrained_unitary_matrix_filename or pretrained_weights_filename) is not None:
            self.eval()

    # Generate orthoganal matrix
    def get_orthogonal_matrix(self, size):
        '''
        Generates an orthoganal matrix of input size
        '''
        # Calculate an orthoganal matrix the size of A
        return torch.nn.init.orthogonal_(torch.empty(size, size))

    # Orthogonal transformation
    def orthogonal_operation(self, input_tensor):
        '''
        input tensor A nxn
        generate orthoganal matrix U the size of (n*n)x(n*n)

        Returns UA
        '''
        # Find batch size and feature map size
        batch_size = input_tensor.size()[0]
        A_side_size = int(input_tensor.size()[2])

        # Determine if U is available
        if self.U == None:
            U = copy(torch.eye(A_side_size**2))
        else:
            U = copy(self.U)

        # Push to GPU if True
        U = copy(U if self.gpu == False else U.cuda())

        # Repeat U and U transpose for all batches
        input_tensor = input_tensor if self.gpu == False else input_tensor.cuda()
        # Ut = U.t().view((1, A_side_size**2, A_side_size**2)).repeat(batch_size, 1, 1)
        U = copy(U.view((1, A_side_size**2, A_side_size**2)).repeat(batch_size, 1, 1))
        
        # Batch muiltply UA
        return torch.bmm(U, input_tensor.view(batch_size, A_side_size**2, 1)).view(batch_size, 1, A_side_size, A_side_size)

    def forward(self, x):
        
        # Unitary transformation
        x = self.orthogonal_operation(x)
        
        # Feedforward
        x = x.view(-1, self.flat_size)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        
        return x

