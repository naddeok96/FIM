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
import xlwt 
from xlwt import Workbook 

# Hyperparameters
gpu = False
set_name = "MNIST"
epsilons = np.round(np.arange(0, 8, 0.2).tolist(), decimals=1)

# Declare which GPU PCI number to use
if gpu == True:
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"

# Initialize table
table = PrettyTable()

# Initialize data
data = Data(gpu, set_name)

# Load LeNet
lenet = AdjLeNet(set_name = set_name)
lenet.load_state_dict(torch.load('models/pretrained/classic_lenet_w_acc_98.pt', map_location=torch.device('cpu')))
lenet.eval()

# # UniConstLeNet
uni_const_lenet = FstLayUniLeNet(set_name = set_name, gpu = gpu)
uni_const_lenet.load_state_dict(torch.load('models/pretrained/mnist_fstlay_uni_const_lenet_w_acc_95.pt', map_location=torch.device('cpu')))
uni_const_lenet.U = torch.load('models/pretrained/U_mnist_fstlay_uni_const_lenet_w_acc_95.pt', map_location=torch.device('cpu'))
uni_const_lenet.eval()

# Enter student network and curriculum data into an academy
academy  = Academy(uni_const_lenet, data, gpu)

# Get model accuracy
test_acc = academy.test()

ossa_accs, fgsm_accs = [], []
for EPSILON in epsilons:
    # Create an attacker
    attacker = OSSA(net = uni_const_lenet, 
                    data = data, 
                    EPSILON = EPSILON,
                    gpu = gpu)

    # Get attack accuracies
    ossa_accs.append((test_acc - attacker.get_attack_accuracy()) / test_acc)
    fgsm_accs.append((test_acc - attacker.get_FGSM_attack_accuracy(uni_net = uni_const_lenet)) / test_acc)
    
table.add_column("Mean", epsilons)
table.add_column("OSSA", ossa_accs)
table.add_column("FGSM", ossa_accs)

print("Test Accuracy: ", test_acc)
print(table)



# Excel Workbook Object is created 
wb = Workbook() 

# Create sheet
sheet = wb.add_sheet('Results') 
sheet.write(0, 0, "Mean")
sheet.write(0, 1, "OSSA")
sheet.write(0, 2, "FGSM")
for i in range(len(epsilons)):
    sheet.write(i + 1, 0, epsilons[i])
    sheet.write(i + 1, 1, ossa_accs[i])
    sheet.write(i + 1, 2, fgsm_accs[i])
wb.save("epsilon_test" + '.xls')