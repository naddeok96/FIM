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
epsilons = np.linspace(0, 12, 61)

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

# UniLeNets

# Initialize
Unet1  = FstLayUniLeNet(set_name = set_name)
Unet2  = FstLayUniLeNet(set_name = set_name)
Unet3  = FstLayUniLeNet(set_name = set_name)
Unet4  = FstLayUniLeNet(set_name = set_name)
Unet5  = FstLayUniLeNet(set_name = set_name)
Unet6  = FstLayUniLeNet(set_name = set_name)
Unet7  = FstLayUniLeNet(set_name = set_name)
Unet8  = FstLayUniLeNet(set_name = set_name)
Unet9  = FstLayUniLeNet(set_name = set_name)
Unet10 = FstLayUniLeNet(set_name = set_name)

# Load Params
Unet1.load_state_dict(torch.load('models/pretrained/U1_of_10_net_w_acc_95.pt', map_location=torch.device('cpu')))
Unet2.load_state_dict(torch.load('models/pretrained/U2_of_10_net_w_acc_95.pt', map_location=torch.device('cpu')))
Unet3.load_state_dict(torch.load('models/pretrained/U3_of_10_net_w_acc_95.pt', map_location=torch.device('cpu')))
Unet4.load_state_dict(torch.load('models/pretrained/U4_of_10_net_w_acc_95.pt', map_location=torch.device('cpu')))
Unet5.load_state_dict(torch.load('models/pretrained/U5_of_10_net_w_acc_95.pt', map_location=torch.device('cpu')))
Unet6.load_state_dict(torch.load('models/pretrained/U6_of_10_net_w_acc_95.pt', map_location=torch.device('cpu')))
Unet7.load_state_dict(torch.load('models/pretrained/U7_of_10_net_w_acc_95.pt', map_location=torch.device('cpu')))
Unet8.load_state_dict(torch.load('models/pretrained/U8_of_10_net_w_acc_95.pt', map_location=torch.device('cpu')))
Unet9.load_state_dict(torch.load('models/pretrained/U9_of_10_net_w_acc_95.pt', map_location=torch.device('cpu')))
Unet10.load_state_dict(torch.load('models/pretrained/U10_of_10_net_w_acc_95.pt', map_location=torch.device('cpu')))

# Load U's
with open("models/pretrained/" + str(0)+ "_of_" + str(10) + "_Us" + '.pkl', 'rb') as input:
    Unet1.U = pickle.load(input).type(torch.FloatTensor)
with open("models/pretrained/" + str(1)+ "_of_" + str(10) + "_Us" + '.pkl', 'rb') as input:
    Unet2.U = pickle.load(input).type(torch.FloatTensor)
with open("models/pretrained/" + str(2)+ "_of_" + str(10) + "_Us" + '.pkl', 'rb') as input:
    Unet3.U = pickle.load(input).type(torch.FloatTensor)
with open("models/pretrained/" + str(3)+ "_of_" + str(10) + "_Us" + '.pkl', 'rb') as input:
    Unet4.U = pickle.load(input).type(torch.FloatTensor)
with open("models/pretrained/" + str(4)+ "_of_" + str(10) + "_Us" + '.pkl', 'rb') as input:
    Unet5.U = pickle.load(input).type(torch.FloatTensor)
with open("models/pretrained/" + str(5)+ "_of_" + str(10) + "_Us" + '.pkl', 'rb') as input:
    Unet6.U = pickle.load(input).type(torch.FloatTensor)
with open("models/pretrained/" + str(6)+ "_of_" + str(10) + "_Us" + '.pkl', 'rb') as input:
    Unet7.U = pickle.load(input).type(torch.FloatTensor)
with open("models/pretrained/" + str(7)+ "_of_" + str(10) + "_Us" + '.pkl', 'rb') as input:
    Unet8.U = pickle.load(input).type(torch.FloatTensor)
with open("models/pretrained/" + str(8)+ "_of_" + str(10) + "_Us" + '.pkl', 'rb') as input:
    Unet9.U = pickle.load(input).type(torch.FloatTensor)
with open("models/pretrained/" + str(9)+ "_of_" + str(10) + "_Us" + '.pkl', 'rb') as input:
    Unet10.U = pickle.load(input).type(torch.FloatTensor)

# Evaluate
Unet1.eval()
Unet2.eval()
Unet3.eval()
Unet4.eval()
Unet5.eval()
Unet6.eval()
Unet7.eval()
Unet8.eval()
Unet9.eval()
Unet10.eval()

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

