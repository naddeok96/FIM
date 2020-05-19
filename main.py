'''
This code will be used as the main code to run all classes
'''
# Imports
import torch
import pickle
from models.classes.adjustable_lenet import AdjLeNet
from models.classes.first_layer_rand_lenet import FstLayRandLeNet
from models.classes.first_layer_unitary_lenet import FstLayUniLeNet
from models.classes.rand_lenet import RandLeNet
from models.classes.unitary_lenet import UniLeNet
from data_setup import Data
from academy import Academy
from ossa import OSSA
import torchvision.transforms.functional as F
import operator
import numpy as np

# Hyperparameters
gpu = False
set_name = "MNIST"
save_set = False
EPSILON = 8
models = {"mnist_lenet_w_acc_98"                : AdjLeNet(set_name = set_name, num_kernels_layer3 = 100),
            "mnist_fstlay_randlenet_w_acc_35"   : FstLayRandLeNet(set_name = set_name, gpu = gpu, num_kernels_layer3 = 100),
            "mnist_fstlay_unilenet_w_acc_30"    : FstLayUniLeNet(set_name = set_name, gpu = gpu, num_kernels_layer3 = 100), 
            "mnist_randlenet_w_acc_97"          : RandLeNet(set_name = set_name, gpu = gpu, num_kernels_layer3 = 100),
            "mnist_unilenet_w_acc_95"           : UniLeNet(set_name = set_name, gpu = gpu, num_kernels_layer3 = 100)}

# Declare which GPU PCI number to use
if gpu == True:
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# Initialize data
data = Data(gpu, set_name)

print("=========================================================")
print("|\t  Model Name\t\t|    Attack Accuracy\t|")
print("|-------------------------------|-----------------------|")
for model_name in models:
    # Load pretraind MNIST network
    net = models[model_name]
    net.load_state_dict(torch.load("models/pretrained/" + model_name + ".pt", map_location=torch.device('cpu')))
    net.eval()

    # Generate Attacks
    ossa = OSSA(net, data, EPSILON = 8, gpu = gpu, model_name=model_name)
    if model_name == "mnist_lenet_w_acc_98": 
        print("|" + str(model_name) + "\t\t|\t " + str(float(round(ossa.get_attack_accuracy() * 100, 2))) + "\t\t|")

    elif model_name == "mnist_fstlay_randlenet_w_acc_35": 
        print("|" + str(model_name) + "|\t " + str(float(round(ossa.get_attack_accuracy() * 100, 2))) + "\t\t|")

    elif model_name == "mnist_fstlay_unilenet_w_acc_30":
        print("|" + str(model_name) + " |\t " + str(float(round(ossa.get_attack_accuracy() * 100, 2))) + "\t\t|")

    else:
        print("|" + str(model_name) + " \t|\t " + str(float(round(ossa.get_attack_accuracy() * 100, 2))) + "\t\t|")

print("=========================================================")

