'''
This code will be used as the main code to run all classes
'''
# Imports
import torch
from torch.autograd import Variable
from unitary_lenet import UniLeNet
from data_setup import Data
from academy import Academy
from information_geometry import InfoGeo
import torchvision.transforms.functional as F
import operator
import numpy as np

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
net = UniLeNet(set_name = set_name,
                num_kernels_layer3 = 100)

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

        # Reshape image and make it a variable which requires a gradient
        image = image.view(1,1,28,28)
        image = Variable(image, requires_grad = True) if gpu == False else Variable(image.cuda(), requires_grad = True)
        
        # Reshape label
        label = torch.tensor([label.item()])

        # Calculate Orginal Loss
        output = net(image)
        

