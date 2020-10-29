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
gpu = True
set_name = "MNIST"

# Declare which GPU PCI number to use
if gpu == True:
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Initialize table
table = PrettyTable()

# Initialize data
data = Data(gpu, set_name)

# Load LeNet
lenet = AdjLeNet(set_name = set_name)
lenet.load_state_dict(torch.load('models/pretrained/classic_lenet_w_acc_98.pt', map_location=torch.device('cpu')))
lenet.eval()

# Generate U
U = torch.nn.init.orthogonal_(torch.empty(784, 784))

# Push to GPU if True
U = U if gpu == False else U.cuda()

# Enter student network and curriculum data into an academy
academy  = Academy(lenet, data, gpu)

# Get model accuracy
test_acc = academy.test()

# Create Attacker
attacker = Attacker(net = lenet, 
                    data = data,
                    gpu = gpu)

# Declare epsilons
epsilons = [x/5 for x in range(41)]
table.add_column("Epsilons", epsilons)

# Get attack accuracies
print("Working on OSSA Attacks...")
ossa_accs  = attacker.get_OSSA_attack_accuracy(epsilons = epsilons,
                                               U = None)
ossa_fool_ratio = attacker.get_fool_ratio(test_acc, ossa_accs)
table.add_column("OSSA Attack Accuracy", ossa_fool_ratio)

print("Working on U(eta) Attacks...")
U_eta_accs = attacker.get_OSSA_attack_accuracy(epsilons = epsilons, 
                                               U = U)
U_eta_fool_ratio = attacker.get_fool_ratio(test_acc, U_eta_accs)
table.add_column("OSSA (Ueta) Attack Accuracy", U_eta_fool_ratio)

print("Working on FGSM Attacks...")
fgsm_accs = attacker.get_FGSM_attack_accuracy(epsilons = epsilons)
fgsm_fool_ratio = attacker.get_fool_ratio(test_acc, fgsm_accs)
table.add_column("FGSM Attack Accuracy", fgsm_fool_ratio)

# Display
print(table)
