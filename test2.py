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
table = PrettyTable()
table.field_names = ["Accuracy"]


# Hyperparameters
gpu = False
set_name = "MNIST"

# Declare which GPU PCI number to use
if gpu == True:
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# Initialize table
table = PrettyTable()

# Initialize data
data = Data(gpu, set_name)

# Load LeNet
lenet = AdjLeNet(set_name = set_name)
lenet.load_state_dict(torch.load('models/pretrained/classic_lenet_w_acc_98.pt', map_location=torch.device('cpu')))
lenet.eval()

# # UniConstLeNet
uni_const_lenet = FstLayUniLeNet(set_name = set_name)
uni_const_lenet.load_state_dict(torch.load('models/pretrained/mnist_fstlay_uni_const_lenet_w_acc_95.pt', map_location=torch.device('cpu')))
uni_const_lenet.U = torch.load('models/pretrained/U_mnist_fstlay_uni_const_lenet_w_acc_95.pt', map_location=torch.device('cpu'))
uni_const_lenet.eval()

# Enter student network and curriculum data into an academy
academy  = Academy(uni_const_lenet, data, gpu)

# Create an attacker
attacker = OSSA(lenet, data)

# # View attack
# num2view = 100
# index = 0
# for _ in range(num2view):
#     image, label, index = data.get_single_image(index = index)
#     _, predicted, adv_predictions = attacker.get_newG_attack(image, label)
#     while ((label != predicted).item()) or (sum(adv_prediction == label for adv_prediction in adv_predictions).item() != 1): # 0 for normal
#         index += 1
#         image, label, index = data.get_single_image(index = index)
#         _, predicted, adv_predictions = attacker.get_newG_attack(image, label)

#     attacker.get_newG_attack(image, label, plot = True)
#     index += 1

# Get model accuracy
table.add_column("Test Accuracy", [academy.test()])

# Get attack accuracy
table.add_column("Attack Accuracy", [attacker.get_UGU_attack_accuracy(uni_net = uni_const_lenet)])
table.add_column("UGU Attack Accuracy", [attacker.get_UGU_attack_accuracy(uni_net = uni_const_lenet, U = uni_const_lenet.U)])
table.add_column("FGSM Attack Accuracy", [attacker.get_FGSM_attack_accuracy(uni_net = uni_const_lenet)])

print(table)