'''
This class builds a NN an optional Unitary Operator on the image
''' 
# Imports
import torch
from torch import nn
from copy import copy
from torch.autograd import Variable
import sys
sys.path.append(".")
from .adjustable_lenet import AdjLeNet

class FstLayUniNet(nn.Module):
    # Initialize
    def __init__(self, set_name,
                       gpu        = False,
                       U_filename = None,
                       model_name = None, 
                       pretrained_accuracy = 100,
                       pretrained_weights_filename = None,
                       desired_image_size = None):

        super(FstLayUniNet,self).__init__()

        # Declare Setname 
        self.set_name = set_name
        if self.set_name == "TinyImageNet":
            self.num_channels = 3
            self.image_size   = 64
            self.num_classes  = 200

        if self.set_name == "CIFAR10":
            self.num_channels = 3
            self.image_size   = 32
            self.num_classes  = 10

        elif self.set_name == "MNIST":
            self.num_channels = 1
            self.image_size   = 28 if desired_image_size is None else desired_image_size
            self.num_classes  = 10

        else:
            print("Please enter a valid set_name (MNIST, CIFAR10 and TinyImageNet).")
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
        if self.set_name == "MNIST":
            if model_name == "lenet":
                
                self.net =  AdjLeNet(set_name   = self.set_name, 
                                    num_classes = self.num_classes,
                                    pretrained_weights_filename = pretrained_weights_filename)

                self.accuracy = pretrained_accuracy
            
            else:
                print("Only lenet is available for MNIST")
                exit()

        elif self.set_name == "CIFAR10":

            self.net = torch.hub.load("chenyaofo/pytorch-cifar-models", self.model_name, pretrained=False, verbose = False)

        elif self.set_name == "TinyImageNet":
            from efficientnet_pytorch import EfficientNet

            if pretrained_weights_filename:
                self.net = EfficientNet.from_pretrained('efficientnet-b7', num_classes = self.num_classes)
            else:
                self.net = EfficientNet.from_name('efficientnet-b7', num_classes = self.num_classes)

        else:
            print("Please enter vaild dataset. (Options: MNIST, CIFAR10, TinyImageNet)")
            exit()

        # Organize GPUs
        if isinstance(self.gpu, bool):
            # Push to GPU if True
            if self.gpu:
                self.net = self.net.cuda() 

        else:
            # Push to rank of gpu
            self.net = self.net.to(gpu)

        # Evaluation Tools 
        self.criterion      = torch.nn.CrossEntropyLoss()
        self.indv_criterion = torch.nn.CrossEntropyLoss(reduction = 'none')
        self.soft_max       = torch.nn.Softmax(dim = 1)
        
    # Generate orthoganal matrix with no Mean or STD Stats
    def get_orthogonal_matrix(self, size):
        '''
        Generates an orthoganal matrix of input size
        '''
        # Calculate an orthoganal matrix the size of A
        return torch.nn.init.orthogonal_(torch.empty(size, size))

    def get_FIM(self, images):
        """Calculate the Fisher Information Matrix for all images

        Args:
            images : Images to be used

        Returns:
            FIM, Loss for each Image, Predicted Class for each image
        """
        # Push to gpu
        if isinstance(self.gpu, bool):
            images = Variable(images, requires_grad = True) if self.gpu == False else Variable(images.cuda(), requires_grad = True)
        else:
            images = Variable(images.to(self.gpu), requires_grad = True)
        
        # Make images require gradients
        images.requires_grad_(True)

        #Forward pass
        outputs         = self.net(images)
        soft_max_output = self.soft_max(outputs)

        # Find size parameters
        batch_size  = outputs.size(0)
        num_classes = outputs.size(1)

        # Calculate FIMs
        fisher = 0 
        for i in range(num_classes):
            # Clear Gradients
            self.net.zero_grad()
            images.grad = None

            # Cycle through lables (y)
            temp_labels = torch.tensor([i]).repeat(batch_size) 
            if isinstance(self.gpu, bool):
                if self.gpu:
                    temp_labels = temp_labels.cuda()
            else:
                temp_labels= temp_labels.to(self.gpu)
            
            # Calculate losses
            temp_loss = self.criterion(outputs, temp_labels)
            temp_loss.backward(retain_graph = True)

            # Calculate expectation
            p    = soft_max_output[:,i].view(batch_size, 1, 1, 1)
            grad = images.grad.data.view(batch_size, 28*28, 1)

            fisher += p * torch.bmm(grad, torch.transpose(grad, 1, 2)).view(batch_size, 1, 28*28, 28*28)
       
        return fisher

    def get_optimal_orthogonal_matrix(self, source_image):
        '''
        Generates an optimal orthoganal matrix of input size
        '''
        fisher_info_mat = self.get_FIM(source_image)
        print(fisher_info_mat.size())
        exit()

        # Orthogonal matrix with first 2 columns as the eigenvectors associated with the min and max eigenvalues
        D = self.GramSchmidt(fisher_info_mat)

        # Permutation matrix
        P = torch.eye(fisher_info_mat.size(0))
        index = torch.tensor(range(P.size(0)))
        index[0:2] = torch.tensor([1, 0])
        P = P[index]

        # Basis change of D from P
        return torch.mm(torch.mm(D,P),torch.transpose(D, 0, 1))

    # GramSchmidt Algorithm
    def GramSchmidt(A):
        # Get eigensystem
        eigenvalues, eigenvectors = torch.linalg.eig(A)

        # Get eigenvectors associated with the min/max eigenvalues
        min_idx = torch.argmin(torch.abs(torch.real(eigenvalues)))
        max_idx = torch.argmax(torch.abs(torch.real(eigenvalues)))
    
        eig_min = eigenvectors[:, min_idx]
        eig_max = eigenvectors[:, max_idx]

        # Generate random matrix to transform into unitary
        V = torch.randn_like(A)

        # Replace first and second column with eigenvectors associated with the min/max eigenvalues
        V[0,:] = torch.real(eig_min)
        V[1,:] = torch.real(eig_max)

        # Orthogonal complement of V in n-dimension 
        for i in range(A.size(0)):
            # Orthonormalize
            Ui = copy.copy(V[i])

            for j in range(i):
                Uj = copy.copy(V[j])

                
                Ui = Ui - ((torch.dot(Uj.view(-1), Ui.view(-1)) / (torch.linalg.norm(Uj, ord = 2)**2)))*Uj
                
            V[i] = Ui / torch.linalg.norm(Ui, ord = 2)
            
        return V.t()

    # Set a new U with no Mean or STD Stats
    def set_orthogonal_matrix(self, source_network = None, source_image = None):
        if source_image:
            self.U = self.get_optimal_orthognal_matrix(source_image)
        else:
            self.U = self.get_orthogonal_matrix(self.image_size)

        # Organize GPUs
        if isinstance(self.gpu, bool):
            # Push to GPU if True
            self.U = self.U.cuda() if self.gpu else self.U

        else:
            # Push to rank of gpu
            self.U = self.U.to(self.gpu)
        
    # Load unitary matrix from file
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
            if sofar == "models/pretrained/" + self.set_name + "/U_w_means_" or sofar == "models/pretrained/" + self.set_name + "/weak_U_w_means_":
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

    # Display summary
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

    # Forward pass
    def forward(self, x):
        
        # Unitary transformation
        x = self.orthogonal_operation(x)

        # Feedforward
        x = self.net(x)

        return x

