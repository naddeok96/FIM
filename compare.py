'''
This code will be used to compare models
'''
# Imports
import torch
from torch.autograd import Variable
from models.classes.adjustable_lenet import AdjLeNet
from models.classes.first_layer_unitary_lenet   import FstLayUniLeNet
from data_setup import Data
from academy import Academy
from ossa import OSSA
import torchvision.transforms.functional as F
import operator
import numpy as np
from prettytable import PrettyTable

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
# LeNet
lenet = AdjLeNet(set_name = set_name)
lenet.load_state_dict(torch.load('models/pretrained/mnist_fstlay_uni_const_lenet_w_acc_95.pt', map_location=torch.device('cpu')))
lenet.eval()

# UniConstLeNet
uni_const_lenet = FstLayUniLeNet(set_name = set_name)
uni_const_lenet.load_state_dict(torch.load('models/pretrained/mnist_fstlay_uni_const_lenet_w_acc_95.pt', map_location=torch.device('cpu')))
uni_const_lenet.U = torch.load('models/pretrained/U_mnist_fstlay_uni_const_lenet_w_acc_95.pt', map_location=torch.device('cpu'))
uni_const_lenet.eval()

# Load into a dictionary
networks = {"lenet":             {"object": lenet,
                                    "eigenvalues"  : [],
                                    "eigenvectors" : []},

            "uni_const_lenet" :  {"object": uni_const_lenet,
                                    "eigenvalues"  : [],
                                    "eigenvectors" : []}}


# Cycle through images
table = PrettyTable()
table.field_names = networks.keys()
num_images = 2
for j in range(num_images):
    # Use just one image
    image, label, _ = data.get_single_image()
    
    # Cycle through networks
    for net in networks:
        # Load network and data into an attacker object
        attacker = OSSA(networks[net]["object"], data)

        # Generate FIMs
        fisher, batch_size, num_classes, losses, predicted = attacker.get_fim(image, label)

        # Determine Eigensystem
        eig_values, eig_vectors = attacker.get_eigens(fisher)

        table.add_column(net, torch.flip(eig_values, [2]).detach().numpy())
        
        # # Save
        # networks[net]["eigenvalues"].append(torch.flip(eig_values, [2])) 
        # networks[net]["eigenvectors"].append(eig_vectors) 

    mixed = torch.cat((networks["lenet"]["eigenvalues"][-1], networks["uni_const_lenet"]["eigenvalues"][-1]), 0)
    mixed = mixed.view(2, 784).detach().numpy()

    table.clear_rows()
    for i in range(np.shape(mixed)[1]):
        table.add_row(list(mixed[:, i]))

    print("Comparing Eigenvalues for image " + str(j + 1) + " of " + str(num_images))
    print(table)
    print("\n\n")

