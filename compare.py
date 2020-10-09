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
from adversarial_attacks import Attacker
import torchvision.transforms.functional as F
import operator
import numpy as np
from prettytable import PrettyTable

# Hyperparameters
gpu = False
set_name = "MNIST"
plot = True
save_set = False
epsilons = [0.25, 0.5, 1, 2.5, 5, 8]

# Declare which GPU PCI number to use
if gpu == True:
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# Initialize data
data = Data(gpu, set_name)

# Load pretraind MNIST network
# Attackers Net
attacker_lenet = AdjLeNet(set_name = set_name)
attacker_lenet.load_state_dict(torch.load('models\pretrained\seed100_lenet_w_acc_98.pt', map_location=torch.device('cpu')))
attacker_lenet.eval()

# LeNet
lenet = AdjLeNet(set_name = set_name)
lenet.load_state_dict(torch.load('models\pretrained\classic_lenet_w_acc_98.pt', map_location=torch.device('cpu')))
lenet.eval()

# UniLeNet
Unet = FstLayUniLeNet(set_name = set_name)
Unet.load_state_dict(torch.load('models/pretrained/mnist_fstlay_uni_const_lenet_w_acc_95.pt', map_location=torch.device('cpu')))
Unet.U = torch.load('models/pretrained/U_mnist_fstlay_uni_const_lenet_w_acc_95.pt', map_location=torch.device('cpu'))
Unet.eval()

# Initialize attacker
attacker = Attacker(attacker_lenet, data)

# Find attack accuracies
ossa_accs_on_Unet = attacker.get_OSSA_attack_accuracy(epsilons = epsilons,
                                                      transfer_network = Unet)
ossa_accs_on_Regnet = attacker.get_OSSA_attack_accuracy(epsilons = epsilons,
                                                        transfer_network = lenet)
fgsm_accs_on_Unet = attacker.get_FGSM_attack_accuracy(epsilons = epsilons, 
                                                      transfer_network = Unet)
fgsm_accs_on_Regnet = attacker.get_FGSM_attack_accuracy(epsilons = epsilons, 
                                                        transfer_network = lenet)

print(ossa_accs_on_Unet)
print(ossa_accs_on_Regnet)
print(fgsm_accs_on_Unet)
print(fgsm_accs_on_Regnet)






# Cycle through images
table = PrettyTable()
num_images = 1
for j in range(num_images):
    # Use just one image
    image, label, _ = data.get_single_image()
    
    # Cycle through networks
    for net_name in networks:
        # Load network and data into an attacker object
        attacker = OSSA(networks[net_name]["object"], data)

        # Generate FIMs
        fisher, batch_size, num_classes, losses, predicted = attacker.get_fim_new(image, label, networks[net_name]["object"].U)

        # # Determine Eigensystem
        eig_values, eig_vectors = attacker.get_eigens(fisher)
        
        # Save
        networks[net_name]["fisher"].append(fisher)
        table.add_column(net_name, torch.flip(eig_values, [2]).view(-1).detach().numpy())
    
    # Display
    print("Comparing Eigenvalues for image " + str(j + 1) + " of " + str(num_images))
    print(table)
    print("\n\n")

G = networks["lenet"]["fisher"][0].view(784, 784)
newG = networks["uni_const_lenet"]["fisher"][0].view(784, 784)
U = networks["uni_const_lenet"]["object"].U

UGU = torch.mm(torch.mm(U, G), U.t())

print("newG == UGU: ", torch.all(newG.eq(UGU)).item())
print("newG:")
print(newG)
print("UGU")
print(UGU)