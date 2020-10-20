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
epsilons = [0, 0.25, 0.5, 1, 2.5, 5, 8]

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
attacker_lenet.load_state_dict(torch.load('models/pretrained/seed100_lenet_w_acc_98.pt', map_location=torch.device('cpu')))
attacker_lenet.eval()

# LeNet
lenet = AdjLeNet(set_name = set_name)
lenet.load_state_dict(torch.load('models/pretrained/classic_lenet_w_acc_98.pt', map_location=torch.device('cpu')))
lenet.eval()

# UniLeNet
Unet = FstLayUniLeNet(set_name = set_name)
Unet.load_state_dict(torch.load('models/pretrained/mnist_fstlay_uni_const_lenet_w_acc_95.pt', map_location=torch.device('cpu')))
Unet.U = torch.load('models/pretrained/U_mnist_fstlay_uni_const_lenet_w_acc_95.pt', map_location=torch.device('cpu'))
Unet.eval()

# Initialize attacker
attacker = Attacker(Unet, data)

# Find attack accuracies
print(epsilons)
ossa_accs_on_Unet = attacker.get_OSSA_attack_accuracy(epsilons = epsilons)
fgsm_accs_on_Unet = attacker.get_FGSM_attack_accuracy(epsilons = epsilons)
print(ossa_accs_on_Unet)
print(fgsm_accs_on_Unet)

# Initialize attacker
attacker = Attacker(lenet, data)
ossa_accs_on_Regnet = attacker.get_OSSA_attack_accuracy(epsilons = epsilons)
fgsm_accs_on_Regnet = attacker.get_FGSM_attack_accuracy(epsilons = epsilons)

print(ossa_accs_on_Regnet)
print(fgsm_accs_on_Regnet)

