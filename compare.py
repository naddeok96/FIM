'''
This code will be used to compare models
'''
# Imports
import torch
from torch.autograd import Variable
from unitary_lenet import UniLeNet
from adjustable_lenet import AdjLeNet
from data_setup import Data
from academy import Academy
from information_geometry import InfoGeo
import torchvision.transforms.functional as F
import operator
import numpy as np

import curses
def move_right(by):
	y,x = curses.getyx()
	curses.move(y - 1, x + by)

# Hyperparameters
gpu = False
set_name = "MNIST"
plot = True
save_set = False
EPSILON = 8

# Declare which GPU PCI number to use
if gpu == True:
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# Initialize data
data = Data(gpu, set_name)

# Load pretraind MNIST network
# LeNet Trainned from seed 1
lenet = AdjLeNet(set_name = set_name,
                 num_kernels_layer3 = 100)
lenet.load_state_dict(torch.load('mnist_lenet_w_acc_98.pt', map_location=torch.device('cpu')))
lenet.eval()

# LeNet Trained from LeNet
le_lenet = AdjLeNet(set_name = set_name,
                 num_kernels_layer3 = 100)
le_lenet.load_state_dict(torch.load('mnist_lenet_w_pretrained_le_kernels_w_acc_98.pt', map_location=torch.device('cpu')))
le_lenet.eval()

# LeNet Trained from UniLeNet
uni_lenet = AdjLeNet(set_name = set_name,
                 num_kernels_layer3 = 100)
uni_lenet.load_state_dict(torch.load('mnist_lenet_w_pretrained_uni_kernels_w_acc_97.pt', map_location=torch.device('cpu')))
uni_lenet.eval()

# UniNet Trained from seed 1
unilenet = UniLeNet(set_name = set_name,
                    gpu = gpu,
                    num_kernels_layer3 = 100)
unilenet.load_state_dict(torch.load('mnist_unilenet_w_acc_95.pt', map_location=torch.device('cpu')))
unilenet.eval()

# UniNet Trained from UniLeNet
uni_unilenet = UniLeNet(set_name = set_name,
                         gpu = gpu,
                         num_kernels_layer3 = 100)
uni_unilenet.load_state_dict(torch.load('mnist_unilenet_w_pretrained_uni_kernels_w_acc_95.pt', map_location=torch.device('cpu')))
uni_unilenet.eval()

# UniNet Trained from LeNet
le_unilenet = UniLeNet(set_name = set_name,
                         gpu = gpu,
                         num_kernels_layer3 = 100)
le_unilenet.load_state_dict(torch.load('mnist_unilenet_w_pretrained_le_kernels_w_acc_35.pt', map_location=torch.device('cpu')))
le_unilenet.eval()

networks = {"lenet"         : {"object": lenet,
                                "eig_val_max" : None,
                                "eig_vec_max" : None},

            "le_lenet"      : {"object": le_lenet,
                                "eig_val_max" : None,
                                "eig_vec_max" : None},

            "uni_lenet"     : {"object": uni_lenet,
                                "eig_val_max" : None,
                                "eig_vec_max" : None},

            "unilenet"      : {"object": unilenet,
                                "eig_val_max" : None,
                                "eig_vec_max" : None},

            "uni_unilenet"  : {"object": uni_unilenet,
                                "eig_val_max" : None,
                                "eig_vec_max" : None},

            "le_unilenet"   : {"object": le_unilenet,
                                "eig_val_max" : None,
                                "eig_vec_max" : None}}


# Evaluation Tools
criterion = torch.nn.CrossEntropyLoss()
soft_max = torch.nn.Softmax(dim = 1)

# Generate Attacks
j = 0
total_tested = 1#len(data.test_set)
for inputs, labels in data.test_loader:
    for image, label in zip(inputs, labels):
        # Break for iterations
        if j >= total_tested:
            break
        j += 1
            
        for net_name in networks:

            # Define network
            net = networks[net_name]["object"]

            # Reshape image and make it a variable which requires a gradient
            image = image.view(1,1,28,28)
            image = Variable(image, requires_grad = True) if gpu == False else Variable(image.cuda(), requires_grad = True)
            
            # Reshape label
            label = torch.tensor([label.item()])

            # Calculate Orginal Loss
            output = net(image)
            soft_max_output = soft_max(output)
            loss = criterion(output, label).item()
            _, predicted = torch.max(soft_max_output.data, 1)

            # Calculate FIM
            fisher = 0 
            for i in range(len(output.data[0])):
                # Cycle through lables (y)
                temp_label = torch.tensor([i]) if gpu == False else torch.tensor([i]).cuda()

                # Reset the gradients
                net.zero_grad()
                image.grad = None

                # Calculate losses
                temp_loss = criterion(output, temp_label)
                temp_loss.backward(retain_graph = True)

                # Calculate expectation
                p = soft_max_output.squeeze(0)[i].item()
                fisher += p * (image.grad.data.view(28*28,1) * torch.t(image.grad.data.view(28*28,1)))

            
            # Highest Eigenvalue and vector
            eig_values, eig_vectors = torch.eig(fisher, eigenvectors = True)
            networks[net_name]["eig_val_max"] = eig_values[0][0].item()
            networks[net_name]["eig_vec_max"] = eig_vectors[:, 0]
            if(net_name == "lenet"):
                print("|Network\t| Highest Eigen Value")
                print("|---------------|----------------------")
                print("|",net_name, "\t|", networks[net_name]["eig_val_max"])
            else:
                print("|",net_name, "\t|", networks[net_name]["eig_val_max"])



